"""
FlexRound -- Learnable Rounding based on Element-wise Division (Lee et al.,
ICML 2023) re-implemented as a TFICQuant *encoder*.

Same contract as every other encoder here (DenseGPTQ / TFIC / ...): consume an
IntegerQuantizedTensorState produced by the base (RTN/AWQ) plus the layer's
trust-region stats, and return corrected *dequantized* weights + a diagnostics
dict. It plugs straight into run_fast.py.

Paper <-> code dictionary
-------------------------
FlexRound replaces the fixed rounding-to-nearest lattice with a learnable,
element-wise *division* rule (Eq. 1-2 of the paper):

        W_hat = s1 * round( W / (s1 . S2 . s3) )                (linear layer)

where every entry of s1, S2, s3 is positive and learnable, s1 is a common
quantization grid size, S2 scales W element-wise, and s3 is a per-output-channel
vector. All three are learned jointly by minimizing the layer reconstruction
error  || W X - W_hat X ||_F^2.

In this pipeline we never see X directly; the calibration is summarized by the
second-moment Gram  G = E[x x^T] = Sigma + mu mu^T  (identical to what the TFIC
encoder uses -- gram backend, keep_sigma=True). Because

        || W X - W_hat X ||_F^2  =  n * Tr( R G R^T ),   R = W_hat - W ,

minimizing the reconstruction error over the calibration set is *exactly*
minimizing  Tr(R G R^T)  in expectation. FlexRound optimizes that quadratic with
Adam, using the straight-through estimator for round(.), matching the reference
implementation (github.com/onliwad101/FlexRound_LRQ, quant/UniformAffine
Quantizer.py). Following that reference we parameterize the positive divisor in
log-space and add per-tensor / per-output-channel offsets so the divisor is

        divisor = exp(delta1 + delta2 + delta3) ,

with  delta1 = log(s1)  (initialized from the base grid size),  delta2  (S2, a
full [C, d] tensor, init 0),  delta3  (s3, a [C, 1] per-output-channel vector,
init 0). delta_i = 0 recovers plain RTN, so training starts exactly at the base.

The grid (scale/zero_point) is inherited from the base pass; FlexRound only
re-optimizes the *grid size* s1 (via delta1) jointly with the rounding, never
the group min/max search. This keeps it composable with any base, exactly like
the other encoders.
"""

from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


class _RoundSTE(torch.autograd.Function):
    """round with a straight-through gradient (identity on backward)."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


class FlexRoundEncoder:
    """
    Gradient-based FlexRound corrector.

    Args:
        iters:      Adam optimization steps per layer (paper uses a few k iters;
                    on a per-layer Gram objective a few hundred already converge).
        lr:         Adam learning rate for the FlexRound parameters.
        use_s3:     include the per-output-channel vector s3 (delta3). Ablation
                    Study 2 of the paper shows it helps; on by default.
        clamp_codes: clamp integer codes to the base's [min_int, max_int] range.
        work_dtype: compute dtype for the optimization (fp32 recommended).
    """

    name = "flexround"

    def __init__(self,
                 iters: int = 300,
                 lr: float = 1e-3,
                 use_s3: bool = True,
                 clamp_codes: bool = True,
                 work_dtype=torch.float32):
        self.iters = int(iters)
        self.lr = float(lr)
        self.use_s3 = bool(use_s3)
        self.clamp_codes = bool(clamp_codes)
        self.work_dtype = work_dtype

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_gram(self, state: IntegerQuantizedTensorState,
                    stats: LayerStats) -> torch.Tensor:
        """G = Sigma + mu mu^T, padded to padded_in_features (padding inert)."""
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features
        assert stats.Sigma is not None, (
            "FlexRound needs a materialized G (gram backend, keep_sigma=True; "
            "register 'flexround' in KEEP_SIGMA / NEED_H in run_fast.py).")
        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        G = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)   # [d,d]
        if pin > d:
            Gp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Gp[:d, :d] = G
            idx = torch.arange(d, pin, device=dev)
            Gp[idx, idx] = torch.diagonal(G).mean()   # padded coords inert, PD
            G = Gp
        return G

    # ------------------------------------------------------------------ #
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        dev = state.scale.device
        wdt = self.work_dtype

        G = self._build_gram(state, stats)                # [pin, pin]

        # ---- fixed base tensors (grid inherited from the base pass) --------
        scale = state.scale.to(wdt)                       # [C, pin] = s1 init
        zp = state.zero_point.to(wdt)                     # [C, pin]
        Wf = state.float_weights.to(wdt)                  # [C, pin] original W (padding-zeroed)
        max_int = float(state.max_int)
        min_int = float(state.min_int)
        C, pin = Wf.shape

        # ---- learnable FlexRound parameters (log-space divisor) -----------
        # delta1 = log(s1): start from the base grid size so we begin at RTN.
        delta1 = torch.log(scale.clamp_min(1e-8)).detach().clone().requires_grad_(True)
        # delta2 = S2 (element-wise), init 0  -> exp offset 1.
        delta2 = torch.zeros_like(Wf).requires_grad_(True)
        params = [delta1, delta2]
        if self.use_s3:
            # delta3 = s3, per-output-channel [C, 1], init 0.
            delta3 = torch.zeros(C, 1, device=dev, dtype=wdt).requires_grad_(True)
            params.append(delta3)
        else:
            delta3 = None

        opt = torch.optim.Adam(params, lr=self.lr)

        # RTN baseline energy for diagnostics: R0 = (round(W/s+zp)-zp)*s - W
        with torch.no_grad():
            q0 = torch.round(Wf / scale + zp).clamp(min_int, max_int)
            R0 = (q0 - zp) * scale - Wf
            e_rtn = (R0 @ G * R0).sum().item()
            del q0, R0

        # ---- optimize Tr(R G R^T) with STE rounding -----------------------
        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)

            log_div = delta1 + delta2
            if delta3 is not None:
                log_div = log_div + delta3
            divisor = torch.exp(log_div)                  # s1 . S2 . s3  > 0
            s1 = torch.exp(delta1)                         # common grid size

            x_int = _round_ste(Wf / divisor)              # round(W / (s1 S2 s3))
            if self.clamp_codes:
                # keep the same code range as the base; zero_point-centred so the
                # clamp bounds match dequantize()'s [min_int, max_int] on codes.
                x_int = torch.clamp(x_int + zp, min_int, max_int) - zp
            W_hat = x_int * s1                            # dequant = q * s1  (paper Eq. 1)

            R = W_hat - Wf                               # [C, pin]
            loss = (R @ G * R).sum()                     # Tr(R G R^T), n-scaled

            loss.backward()
            opt.step()

        # ---- commit corrected weights (no grad) ---------------------------
        with torch.no_grad():
            log_div = delta1 + delta2
            if delta3 is not None:
                log_div = log_div + delta3
            divisor = torch.exp(log_div)
            s1 = torch.exp(delta1)
            x_int = torch.round(Wf / divisor)
            if self.clamp_codes:
                x_int = torch.clamp(x_int + zp, min_int, max_int) - zp
            W_hat = x_int * s1
            Rf = W_hat - Wf                              # exact final energy
            e_flex = (Rf @ G * Rf).sum().item()

            corrected = W_hat
            if state.padded_in_features > state.in_features:
                corrected = corrected[:, : state.in_features]
            corrected = corrected.to(state.original_dtype)

        diag = {
            "encoder": "flexround",
            "iters": self.iters,
            "lr": self.lr,
            "use_s3": self.use_s3,
            "energy_rtn": e_rtn,
            "energy_flex": e_flex,
            "energy_drop": e_rtn - e_flex,
        }
        del G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return corrected, diag
