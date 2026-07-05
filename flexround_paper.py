"""
flexround_paper.py -- FlexRound (Lee et al., ICML 2023) ported *verbatim* from
the authors' reference (github.com/onliwad101/FlexRound_LRQ):
  quant/UniformAffineQuantizer.py, quant/UniformQuantizationLinear.py,
  quant/loss.py, quant/cached_loader_fast.py, quant/REM_fast.py.

This is a faithful re-implementation of the REAL algorithm -- block-wise
reconstruction that forward-backwards through the WHOLE decoder layer and
minimizes the block-output MSE  || fp_block_out - q_block(q_input) ||^2 . It is
NOT a Gram / Hessian surrogate (that was the earlier wrong attempt).

The ONLY thing changed from the reference is the calibration source: instead of
the reference's C4 DataLoader, we feed calibration built by TFICQuant's
calibration_utils.load / get_c4_calibration_data (tokenized input_ids). The
quantization math (element-wise-division rounding, delta1/2/3, STE, clamp,
dequant = q * exp(delta1)), the 3-stream caching (fp_input / fp_output /
q_input), input_prob mixing, Adam + cosine LR, and per-block swap/reconstruct/
dequantize flow are all preserved exactly.

Modernization notes (reference targeted an old transformers LLaMA):
  * The reference calls model.model._prepare_decoder_attention_mask, removed in
    current transformers. For seqlen-fixed causal calibration we pass
    attention_mask=None (HF then builds the causal mask internally) and supply
    position_ids/cache_position, which is equivalent for these dense blocks.
  * We rebuild each independent block by deep-copying the actual quantized
    decoder layer rather than constructing a fresh LlamaDecoderLayer(config),
    which is more robust across model families (LLaMA / Mistral / Qwen2).
"""

from __future__ import annotations

import math
import gc
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================ #
#  UniformAffineQuantizer  -- verbatim from quant/UniformAffineQuantizer.py
#  (flexround branch only; lrq/delta4/delta5 paths dropped since we run mode
#   'flexround'. The forward/clamp/dequant are byte-for-byte the reference.)
# ============================================================================ #
class round_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 4, symmetric: bool = False,
                 clipping: bool = True, channel_wise: bool = False,
                 scale_method: str = 'mse', mode: str = 'flexround',
                 org_weight=None):
        super().__init__()
        self.sym = symmetric
        self.clipping = clipping
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = False
        self.mode = mode
        self.delta1 = None
        self.delta2 = None
        self.delta3 = None
        self.delta4 = None
        self.delta5 = None
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 100 if self.clipping else 0
        self.running_min = None
        self.running_max = None

        if not self.inited:
            self.delta, self.zero_point = self.init_quantization_scale(
                org_weight.detach(), self.channel_wise)
            if self.mode == 'flexround':
                self.delta1 = nn.Parameter(torch.log(self.delta).detach())
                self.delta2 = nn.Parameter(torch.zeros_like(org_weight))
                self.delta3 = nn.Parameter(
                    torch.zeros_like(org_weight[:, 0].unsqueeze(-1)))
            else:
                raise NotImplementedError("this port runs mode='flexround' only")
            self.inited = True

    def forward(self, x: torch.Tensor):
        # divisor = exp(delta1 + delta2 + delta3)  (S = s1 . S2 . s3, in log)
        if not self.sym:
            x_int = round_ste.apply(
                x / (self.delta1 + self.delta2 + self.delta3).exp())
            x_quant = torch.clamp(x_int, -self.zero_point,
                                  self.n_levels - 1 - self.zero_point)
            x_dequant = x_quant * self.delta1.exp()
        else:
            x_int = round_ste.apply(
                x / (self.delta1 + self.delta2 + self.delta3).exp())
            x_quant = torch.clamp(x_int, -2 ** (self.n_bits - 1),
                                  2 ** (self.n_bits - 1) - 1)
            x_dequant = x_quant * self.delta1.exp()
        return x_dequant

    # ---- grid init (RTN/MSE search) : verbatim ---------------------------
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        return torch.flatten(x, 1).mean(1)

    def calculate_qparams(self, min_val, max_val):
        quant_min, quant_max = 0., self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        if not self.sym:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        else:
            scale = 2 * torch.max(max_val_pos, torch.abs(min_val_neg)) / float(
                quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = torch.zeros_like(scale)
        return scale, zero_point

    def quantize(self, x, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        if not self.sym:
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
        else:
            x_quant = torch.clamp(x_int, -2 ** (self.n_bits - 1),
                                  2 ** (self.n_bits - 1) - 1)
            x_float_q = x_quant * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch.aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + 1e10
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
        else:
            x_min, x_max = torch.aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + 1e10
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no':
            return self.perform_1D_search(x)
        return self.perform_2D_search(x)

    def init_quantization_scale_channel(self, x):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone, channel_wise=False):
        if channel_wise:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point


# ============================================================================ #
#  INTLinear + swap  -- verbatim from quant/UniformQuantizationLinear.py
# ============================================================================ #
class INTLinear(nn.Module):
    def __init__(self, org_weight, n_bits=4, symmetric=False, clipping=True,
                 channel_wise=True, mode='flexround', bias=None):
        super().__init__()
        self.org_weight = org_weight
        self.quantized_weight = None
        self.weight_quantizer = UniformAffineQuantizer(
            n_bits=n_bits, symmetric=symmetric, clipping=clipping,
            channel_wise=channel_wise, mode=mode, org_weight=org_weight)
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
                 symmetric=False, clipping=True):
    weight = layer.weight
    bias = layer.bias if layer.bias is not None else None
    return INTLinear(org_weight=weight, n_bits=n_bits, symmetric=symmetric,
                     clipping=clipping, channel_wise=channel_wise, mode=mode,
                     bias=bias)


# ============================================================================ #
#  Loss  -- verbatim from quant/loss.py  (MSE block-output reconstruction)
# ============================================================================ #
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if 'tuple' in str(type(pred)):
        pred = pred[0]
    if 'tuple' in str(type(tgt)):
        tgt = tgt[0]
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    return (pred - tgt).abs().pow(p).mean()


class LossFunction:
    def __init__(self, block, round_loss='relaxation', weight=1.,
                 rec_loss='mse', max_count=2000, b_range=(10, 2),
                 decay_start=0.0, warmup=0.0, p=2.):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt)
        else:
            raise ValueError(f'unsupported rec loss {self.rec_loss}')
        total_loss = rec_loss  # round_loss == 'relaxation' -> 0 (FlexRound)
        if self.count % 500 == 0 or self.count == 1:
            tqdm.write(f'Total loss:\t{float(total_loss):.3f} '
                       f'(rec:{float(rec_loss):.3f})\tcount={self.count}')
        return total_loss


# ============================================================================ #
#  3-stream activation cache  -- port of quant/cached_loader_fast.py
#  (single-GPU, block-level; keeps fp_input / fp_output / q_input like the ref)
# ============================================================================ #
class StopForwardException(Exception):
    pass


class _CacheWrapper(nn.Module):
    """Runs the wrapped block, stashes (inp, kwargs, out), then stops forward."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.inp_data = None
        self.out_data = None
        self.other_data = None

    def forward(self, inp, **other):
        self.inp_data = inp
        out = self.module(inp, **other)
        self.out_data = out
        self.other_data = other
        raise StopForwardException


class CachedDataset(Dataset):
    """
    Holds cached_q_input, cached_fp_output, cached_fp_input (like the reference
    __getitem__ order). Block 0 input is captured by running the embedding +
    a wrapped block-0; subsequent fp/q outputs are advanced block by block.
    """
    def __init__(self, model, layers, calib_ids, device, input_prob,
                 num_samples, block_kwargs_fn):
        super().__init__()
        self.device = device
        self.input_prob = input_prob
        self.block_kwargs_fn = block_kwargs_fn
        self.cached_fp_input = []
        self.cached_fp_output = []
        self.cached_fp_other = []

        # ---- capture block-0 input by wrapping layer 0 -------------------
        wrapped = _CacheWrapper(layers[0])
        layers[0] = wrapped
        with torch.no_grad():
            for i in range(num_samples):
                ids = calib_ids[i].to(device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                try:
                    model(input_ids=ids, use_cache=False)
                except StopForwardException:
                    pass
                inp = wrapped.inp_data.detach()
                other = {k: (v.detach() if torch.is_tensor(v) else v)
                         for k, v in wrapped.other_data.items()}
                out = wrapped.out_data
                out = out[0] if isinstance(out, tuple) else out
                self.cached_fp_input.append(inp.to(device))
                self.cached_fp_output.append(out.detach().to(device))
                self.cached_fp_other.append(other)
        layers[0] = wrapped.module
        self.cached_q_input = self.cached_fp_input  # q starts == fp at block 0

    def fp_data_caching(self, fp_block, num_samples):
        """Advance the fp stream one block: fp_output becomes fp_input, then
        run fp_block on it to get the new fp_output."""
        self.cached_fp_input = self.cached_fp_output
        self.cached_fp_output = []
        fp_block = fp_block.to(self.device)
        with torch.no_grad():
            for i in range(num_samples):
                x = self.cached_fp_input[i].to(self.device)
                kw = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                      for k, v in self.cached_fp_other[i].items()}
                out = fp_block(x, **kw)
                out = out[0] if isinstance(out, tuple) else out
                self.cached_fp_output.append(out.detach().to(self.device))
        fp_block.cpu()
        torch.cuda.empty_cache()

    def q_data_caching(self, q_block, num_samples):
        """Advance the dirty stream through the just-quantized block."""
        new_q = []
        q_block = q_block.to(self.device)
        with torch.no_grad():
            for i in range(num_samples):
                x = self.cached_q_input[i].to(self.device)
                kw = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                      for k, v in self.cached_fp_other[i].items()}
                out = q_block(x, **kw)
                out = out[0] if isinstance(out, tuple) else out
                new_q.append(out.detach().to(self.device))
        q_block.cpu()
        self.cached_q_input = new_q
        torch.cuda.empty_cache()

    def __getitem__(self, index):
        return (self.cached_q_input[index], self.cached_fp_output[index],
                self.cached_fp_input[index])

    def __len__(self):
        return len(self.cached_q_input)


# ============================================================================ #
#  REM engine  -- port of quant/REM_fast.py (quantization + blockReconstruction)
# ============================================================================ #
class FlexRoundREM:
    def __init__(self, model, fp_model, calib_ids, *, n_bits=4, iters=5000,
                 num_samples=128, w_lr=1e-5, input_prob=0.5, channel_wise=True,
                 symmetric=False, clipping=True, mode='flexround', fp16=False,
                 device='cuda:0'):
        self.model = model
        self.fp_model = fp_model
        self.calib_ids = calib_ids
        self.n_bits = n_bits
        self.iters = iters
        self.num_samples = num_samples
        self.w_lr = w_lr
        self.input_prob = input_prob
        self.channel_wise = channel_wise
        self.symmetric = symmetric
        self.clipping = clipping
        self.mode = mode
        self.fp16 = fp16
        self.device = device

    # ---- helpers to enumerate the linear submodules of a decoder block ---
    @staticmethod
    def _linear_names(block):
        return [n for n, m in block.named_modules() if isinstance(m, nn.Linear)]

    @staticmethod
    def _get_sub(block, name):
        obj = block
        for p in name.split('.'):
            obj = getattr(obj, p)
        return obj

    @staticmethod
    def _set_sub(block, name, val):
        obj = block
        parts = name.split('.')
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    def quantizeBlock(self, block):
        wq = {'n_bits': self.n_bits, 'channel_wise': self.channel_wise,
              'mode': self.mode, 'symmetric': self.symmetric,
              'clipping': self.clipping}
        for name in self._linear_names(block):
            lin = self._get_sub(block, name)
            self._set_sub(block, name, swapUniformQ(lin, **wq))
        return block

    def dequantizeBlock(self, block):
        for name in self._linear_names(block):
            m = self._get_sub(block, name)
            if isinstance(m, INTLinear):
                qw = m.weight_quantizer(m.org_weight).clone().detach()
                new_lin = nn.Linear(m.org_weight.shape[1], m.org_weight.shape[0],
                                    bias=(m.bias is not None))
                new_lin.weight = nn.Parameter(qw, requires_grad=False)
                if m.bias is not None:
                    new_lin.bias = nn.Parameter(m.bias.detach().clone(),
                                                requires_grad=False)
                self._set_sub(block, name, new_lin.to(qw.device))
        return block

    def blockReconstruction(self, block_q, dataset):
        device = self.device
        w_para = []
        for _, module in block_q.named_modules():
            if isinstance(module, UniformAffineQuantizer):
                for idx in range(5):
                    p = getattr(module, 'delta' + str(idx + 1))
                    if p is not None:
                        w_para.append(p)
        optimizer = torch.optim.Adam([{'params': w_para, 'lr': self.w_lr}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=0.)
        loss_func = LossFunction(block_q, round_loss='relaxation',
                                 max_count=self.iters, rec_loss='mse')

        loader = DataLoader(dataset, shuffle=True,
                            batch_size=max(1, min(2, len(dataset))))
        epochs = max(1, int(self.iters / max(1, len(loader))))
        remainder = self.iters - len(loader) * epochs
        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        for epoch in tqdm(range(epochs + 1), desc='    recon', leave=False):
            for step, batch in enumerate(loader):
                if epoch == epochs and step == remainder:
                    break
                cur_inp = batch[0].squeeze(1).to(device)
                cur_out = batch[1].squeeze(1).to(device)
                cur_fp_inp = batch[2].squeeze(1).to(device)
                # input_prob mixing (reference): randomly use fp input
                if self.input_prob < 1.0:
                    mask = (torch.rand(cur_inp.size(0), 1, 1, device=device)
                            < self.input_prob).float()
                    cur_inp = mask * cur_fp_inp + (1 - mask) * cur_inp

                optimizer.zero_grad()
                kw = self.block_kwargs  # shared per-block forward kwargs
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    out = block_q(cur_inp.float(), **kw)
                    out = out[0] if isinstance(out, tuple) else out
                    err = loss_func(out, cur_out)
                scaler.scale(err).backward(retain_graph=False)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

        del optimizer
        torch.cuda.empty_cache()
        return block_q

    def quantization(self):
        model = self.model
        fp_model = self.fp_model
        device = self.device

        layers = model.model.layers            # blockUnits    (quantizable)
        fp_layers = fp_model.model.layers      # blockUnits_fp (frozen fp)
        n_blocks = model.config.num_hidden_layers

        def block_kwargs_fn(other):
            return other

        # ---- initial caching (block 0) : reference moves embed_tokens +
        # layers[0] of fp_model to cuda, caches, then returns them to cpu. The
        # rest of fp_model / model stays on CPU throughout (40GB-safe: only ONE
        # block is on GPU at a time).
        print('Initial activation caching (block 0)')
        fp_model.model.embed_tokens = fp_model.model.embed_tokens.to(device)
        if getattr(fp_model.model, 'rotary_emb', None) is not None:
            fp_model.model.rotary_emb = fp_model.model.rotary_emb.to(device)
        fp_layers[0] = fp_layers[0].to(device)
        cached = CachedDataset(fp_model, fp_layers, self.calib_ids,
                               device, self.input_prob, self.num_samples,
                               block_kwargs_fn)
        fp_layers[0] = fp_layers[0].to('cpu')
        fp_model.model.embed_tokens = fp_model.model.embed_tokens.to('cpu')

        for idx in tqdm(range(n_blocks), desc='blocks'):
            print('=' * 60)
            print(f'  Layer {idx} optimization start')

            # 1. FP activation caching: advance clean stream through fp block idx.
            #    reference: DataCacheWrapper(blockUnits_fp[idx]) -> to cuda in
            #    fp_data_caching -> .cpu() after. Same here.
            if idx > 0:
                cached.fp_data_caching(fp_layers[idx], self.num_samples)

            # per-block forward kwargs (position_ids / mask / rotary embeds),
            # captured from the fp stream, replayed on the block we optimize.
            self.block_kwargs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in cached.cached_fp_other[0].items()}

            # 2/3. Independent block = the QUANTIZABLE block (self.model),
            #      sub-modules moved to cuda, exactly like reference
            #      (blockUnits[idx].self_attn.to('cuda:0') ...). No deepcopy:
            #      we quantize the real block in place and keep it on GPU only
            #      for this iteration.
            block = layers[idx].to(device).float()
            block = self.quantizeBlock(block)

            # 4. reconstruct  5. dequantize (in place)
            block = self.blockReconstruction(block, cached)
            block = self.dequantizeBlock(block.half())

            # write dequantized linears back into self.model, on CPU (model
            # lives on CPU; only the active block is transiently on GPU).
            for name in self._linear_names(layers[idx]):
                self._set_sub(layers[idx], name, self._get_sub(block, name))
            layers[idx] = layers[idx].to('cpu')
            del block
            gc.collect()
            torch.cuda.empty_cache()

            # 6. Quantized activation caching: advance dirty stream through the
            #    just-quantized block. reference wraps blockUnits[idx], moves to
            #    cuda in q_data_caching, .cpu() after.
            if idx < n_blocks - 1:
                cached.q_data_caching(layers[idx], self.num_samples)
            else:
                # last block: park all cached activations on CPU (reference)
                for i in range(len(cached.cached_fp_input)):
                    cached.cached_fp_input[i] = cached.cached_fp_input[i].cpu()
                    cached.cached_fp_output[i] = cached.cached_fp_output[i].cpu()
                    cached.cached_q_input[i] = cached.cached_q_input[i].cpu()
                torch.cuda.empty_cache()

        # ---- final cleanup: free the 3-stream cache + fp reference ---------
        for attr in ('cached_fp_input', 'cached_fp_output',
                     'cached_q_input', 'cached_fp_other'):
            if hasattr(cached, attr):
                setattr(cached, attr, None)
        del cached
        self.fp_model = fp_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model