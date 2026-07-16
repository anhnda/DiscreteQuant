"""
flexround_paper_gw.py -- GROUP-WISE variant of the paper-exact FlexRound
(flexround_paper.py). The ONLY change vs flexround_paper.py is the quantization
*grid*: instead of per-output-channel weight quantization, this uses group-wise
asymmetric weight quantization (groups of `group_size` along the input dim), the
same lattice GPTQ/AWQ/RTN-g128 use. Everything else -- the block-wise
reconstruction that forward-backprops through the WHOLE decoder layer, the
3-stream cache, Adam+cosine, the element-wise-division rounding, STE, and the
dequantize-and-bake flow -- is REUSED VERBATIM from flexround_paper.py.

Why a separate file: flexround_paper.py's UniformAffineQuantizer computes a
per-channel grid (one scale/zero_point per output row). For a fair comparison
against the Gram-encoder path (eigenflip/run_fast.py --encoder flexround, which
runs group-wise g=128), we need FlexRound's block-wise reconstruction on the
SAME group-wise grid. This file swaps only that grid.

Paper <-> code mapping (unchanged from flexround_paper.py):
    W_hat = s1 * round( W / (s1 . S2 . s3) )           (Eq. 2, FlexRound)
    delta1 = log(s1)  (grid size, init from RTN),  delta2 = S2 (full, init 0),
    delta3 = s3 (per-output-channel, init 0).  divisor = exp(delta1+delta2+delta3).

Group-wise specifics:
    * grid (scale s1_init, zero_point) is computed per (row, group) block of
      `group_size` input columns, then EXPANDED to [Cout, Cin] so the rounding
      forward is elementwise and identical in shape to the per-channel case.
    * delta1 (log s1) is therefore [Cout, n_groups] expanded to [Cout, Cin]:
      one learnable grid size per (row, group). dequant = q * exp(delta1) is
      per-(row,group), exactly what a group-wise deployed kernel would apply.
    * delta2 (S2) stays full [Cout, Cin]; delta3 (s3) stays [Cout, 1].
    * zero_point is the group-wise asymmetric zp, expanded to [Cout, Cin]; the
      asymmetric clamp uses it elementwise (broadcast already materialized).

At save time dequantizeBlock bakes W_hat exactly like the per-channel port, so
the checkpoint is a normal fp16 Llama checkpoint (fake/simulated quant, for PPL
eval) -- this file does NOT emit packed int+scale deploy artifacts, same as the
original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the ENTIRE block-wise engine verbatim. We only redefine the quantizer,
# the INTLinear wrapper, swapUniformQ, and subclass the REM to thread group_size.
from flexround_paper import (
    round_ste,
    LossFunction,
    lp_loss,
    StopForwardException,
    _CacheWrapper,
    CachedDataset,
    _cast_kwargs,
    FlexRoundREM as _FlexRoundREM_PerChannel,
)


# ============================================================================ #
#  Group-wise UniformAffineQuantizer.
#  Same rounding math as flexround_paper.py, but the grid (scale/zero_point) and
#  the learnable grid size delta1 are computed group-wise and expanded to [C,d].
# ============================================================================ #
class UniformAffineQuantizerGW(nn.Module):
    def __init__(self, n_bits: int = 4, symmetric: bool = False,
                 clipping: bool = True, channel_wise: bool = True,
                 scale_method: str = 'minmax', mode: str = 'flexround',
                 group_size: int = 128, org_weight=None):
        super().__init__()
        self.sym = symmetric
        self.clipping = clipping
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.mode = mode
        self.group_size = group_size
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        assert org_weight is not None and org_weight.dim() == 2, \
            "group-wise quantizer expects a 2D linear weight [Cout, Cin]"
        C, in_features = org_weight.shape
        self.in_features = in_features
        gs = group_size if group_size > 0 else in_features
        gs = min(gs, in_features)
        self.gs = gs
        self.n_groups = (in_features + gs - 1) // gs
        self.padded_in = self.n_groups * gs

        if self.mode != 'flexround':
            raise NotImplementedError("this port runs mode='flexround' only")

        # ---- group-wise asymmetric grid (RTN min/max per (row, group)) ------
        delta_exp, zp_exp = self._init_group_grid(org_weight.detach())
        # delta1 = log(s1): [Cout, padded_in] (expanded per-(row,group))
        self.delta1 = nn.Parameter(torch.log(delta_exp).detach())
        # S2: full [Cout, padded_in], init 0
        self.delta2 = nn.Parameter(torch.zeros(C, self.padded_in,
                                               dtype=org_weight.dtype,
                                               device=org_weight.device))
        # s3: per-output-channel [Cout, 1], init 0
        self.delta3 = nn.Parameter(torch.zeros(C, 1, dtype=org_weight.dtype,
                                               device=org_weight.device))
        # zero_point expanded to [Cout, padded_in] (buffer, not learned)
        self.register_buffer('zero_point_exp', zp_exp)

    # ---- helpers ---------------------------------------------------------
    def _pad(self, W):
        C, in_f = W.shape
        if self.padded_in > in_f:
            Wp = torch.zeros(C, self.padded_in, dtype=W.dtype, device=W.device)
            Wp[:, :in_f] = W
            return Wp
        return W

    @torch.no_grad()
    def _init_group_grid(self, W):
        """Group-wise asymmetric RTN grid. Returns (scale_exp, zp_exp), each
        [Cout, padded_in], replicated within each group of size gs."""
        Wp = self._pad(W)
        C = Wp.shape[0]
        Wg = Wp.reshape(C, self.n_groups, self.gs)
        wmin = Wg.min(dim=2, keepdim=True)[0]
        wmax = Wg.max(dim=2, keepdim=True)[0]
        max_int = self.n_levels - 1
        if not self.sym:
            wmin_neg = torch.min(wmin, torch.zeros_like(wmin))
            wmax_pos = torch.max(wmax, torch.zeros_like(wmax))
            scale_g = ((wmax_pos - wmin_neg) / float(max_int)).clamp_min(1e-8)
            zp_g = torch.clamp(torch.round(-wmin_neg / scale_g), 0, max_int)
        else:
            scale_g = (2 * torch.max(wmax.abs(), wmin.abs())
                       / float(max_int)).clamp_min(1e-8)
            zp_g = torch.zeros_like(scale_g)
        # expand [C, n_groups, 1] -> [C, padded_in]
        scale_exp = scale_g.repeat(1, 1, self.gs).reshape(C, self.padded_in)
        zp_exp = zp_g.repeat(1, 1, self.gs).reshape(C, self.padded_in)
        return scale_exp, zp_exp

    # ---- forward: identical rounding to flexround_paper.py, group grid ----
    def forward(self, x: torch.Tensor):
        # x is org_weight [Cout, Cin]; pad to [Cout, padded_in] so all the
        # [Cout, padded_in] params broadcast elementwise, then strip back.
        xin = x
        if self.padded_in > self.in_features:
            xin = self._pad(x)
        divisor = (self.delta1 + self.delta2 + self.delta3).exp()
        x_int = round_ste.apply(xin / divisor)
        if not self.sym:
            x_quant = torch.clamp(x_int, -self.zero_point_exp,
                                  self.n_levels - 1 - self.zero_point_exp)
        else:
            x_quant = torch.clamp(x_int, -2 ** (self.n_bits - 1),
                                  2 ** (self.n_bits - 1) - 1)
        x_dequant = x_quant * self.delta1.exp()
        if self.padded_in > self.in_features:
            x_dequant = x_dequant[:, :self.in_features]
        return x_dequant

    @torch.jit.export
    def extra_repr(self):
        return ('bit={}, mode={}, group_size={}, n_groups={}, symmetric={}, '
                'clipping={}'.format(self.n_bits, self.mode, self.gs,
                                     self.n_groups, self.sym, self.clipping))


# ============================================================================ #
#  INTLinear + swap  (group-wise variant)
# ============================================================================ #
class INTLinear(nn.Module):
    def __init__(self, org_weight, n_bits=4, symmetric=False, clipping=True,
                 channel_wise=True, mode='flexround', group_size=128,
                 bias=None):
        super().__init__()
        self.org_weight = org_weight
        self.quantized_weight = None
        self.weight_quantizer = UniformAffineQuantizerGW(
            n_bits=n_bits, symmetric=symmetric, clipping=clipping,
            channel_wise=channel_wise, mode=mode, group_size=group_size,
            org_weight=org_weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        if self.quantized_weight is None:
            weight = self.weight_quantizer(self.org_weight)
        else:
            weight = self.quantized_weight
        return F.linear(inputs, weight, self.bias)


def swapUniformQ(layer, n_bits, channel_wise=True, mode='flexround',
                 symmetric=False, clipping=True, group_size=128):
    weight = layer.weight
    bias = layer.bias if layer.bias is not None else None
    return INTLinear(org_weight=weight, n_bits=n_bits, symmetric=symmetric,
                     clipping=clipping, channel_wise=channel_wise, mode=mode,
                     group_size=group_size, bias=bias)


# ============================================================================ #
#  REM engine: reuse everything; override only quantizeBlock (to pass
#  group_size + use the group-wise swap) and dequantizeBlock (INTLinear type).
# ============================================================================ #
class FlexRoundREM(_FlexRoundREM_PerChannel):
    def __init__(self, *args, group_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def quantizeBlock(self, block):
        wq = {'n_bits': self.n_bits, 'channel_wise': self.channel_wise,
              'mode': self.mode, 'symmetric': self.symmetric,
              'clipping': self.clipping, 'group_size': self.group_size}
        for name in self._linear_names(block):
            lin = self._get_sub(block, name)
            self._set_sub(block, name, swapUniformQ(lin, **wq))
        return block

    def dequantizeBlock(self, block, out_dtype=torch.float16):
        # Same bake-to-nn.Linear logic as the per-channel port, but keyed on the
        # group-wise INTLinear defined in THIS module.
        for name in [n for n, m in block.named_modules()
                     if isinstance(m, INTLinear)]:
            m = self._get_sub(block, name)
            qw = m.weight_quantizer(m.org_weight).clone().detach().to(out_dtype)
            new_lin = nn.Linear(qw.shape[1], qw.shape[0],
                                bias=(m.bias is not None))
            new_lin.weight = nn.Parameter(qw, requires_grad=False)
            if m.bias is not None:
                new_lin.bias = nn.Parameter(
                    m.bias.detach().clone().to(out_dtype), requires_grad=False)
            self._set_sub(block, name, new_lin.to(qw.device))
        block = block.to(out_dtype)
        return block

    def blockReconstruction(self, block_q, dataset):
        # The parent collects learnable params by scanning for the ORIGINAL
        # UniformAffineQuantizer type. Our block holds UniformAffineQuantizerGW,
        # so we re-implement the param collection to target it, then delegate to
        # the parent's optimization loop via a tiny shim: easiest is to call the
        # parent but it filters by the wrong class. Instead, replicate the loop
        # here using the same hyperparameters.
        device = self.device
        block_dtype = next(block_q.parameters()).dtype
        w_para = []
        for _, module in block_q.named_modules():
            if isinstance(module, UniformAffineQuantizerGW):
                for idx in range(3):
                    p = getattr(module, 'delta' + str(idx + 1))
                    if p is not None:
                        w_para.append(p)
        optimizer = torch.optim.Adam([{'params': w_para, 'lr': self.w_lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=0.)
        loss_func = LossFunction(block_q, round_loss='relaxation',
                                 max_count=self.iters, rec_loss='mse')

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, shuffle=True,
                            batch_size=max(1, min(2, len(dataset))))
        epochs = max(1, int(self.iters / max(1, len(loader))))
        remainder = self.iters - len(loader) * epochs
        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        from tqdm import tqdm
        for epoch in tqdm(range(epochs + 1), desc='    recon(gw)', leave=False):
            for step, batch in enumerate(loader):
                if epoch == epochs and step == remainder:
                    break
                cur_inp = batch[0].squeeze(1).to(device)
                cur_out = batch[1].squeeze(1).to(device)
                optimizer.zero_grad()
                kw = self._prep_kwargs(self.block_kwargs, cur_inp.size(0),
                                       block_dtype, device)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    out = block_q(cur_inp.to(block_dtype), **kw)
                    out = out[0] if isinstance(out, tuple) else out
                    err = loss_func(out, cur_out.to(out.dtype))
                scaler.scale(err).backward(retain_graph=True)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

        del optimizer
        torch.cuda.empty_cache()
        return block_q
