"""
FlexRound (paper-exact / sequential) -- Learnable Rounding based on Element-wise
Division, Lee et al., ICML 2023 -- as a TFICQuant *asymmetric* encoder.

Difference vs. flexround.py (the layer-wise variant)
----------------------------------------------------
flexround.py minimizes  || W X - W_hat X ||^2  with X the *full-precision*
input (symmetric collect_fast Gram). That is an AdaRound-style, layer-wise
objective -- NOT what the FlexRound paper optimizes.

The paper (Sec. 3.1-3.2) optimizes the *sequential* block/layer objective

        min_{s1, S2, s3}  || W X  -  W_hat X~ ||_F^2                 (paper Eq. 2)

where X is the clean input (all previous layers intact) and X~ (X-tilde) is the
input when *all previous layers are already quantized*. The quantization error
of earlier layers is propagated in through X~, and the current layer must
compensate for it. This is exactly the BRECQ/QDrop reconstruction FlexRound
inherits ("B + / Q +" columns in the paper).

This pipeline already materializes X~ via the block-causal collector
statistics/collect_asym.py (the GPTAQ path): it keeps a clean fp cascade and a
dirty quantized cascade side by side, quantizing each block before advancing.
It hands each layer:

    stats.Sigma  ->  G = Sigma + mu mu^T = (1/n) X~ X~^T   (H built on DIRTY X~)
    stats.F      ->  K = (1/n) A dA^T  = (1/n) X~ (X~ - X)^T-in-input-space
                     (the GPTAQ cross-Gram; add_cross folds  sum_t xd (x~-xd)^T)

Expanding the paper objective with R = W_hat - W and dX = X~ - X:

        || W X - W_hat X~ ||^2
      = || -R X~ - W dX ||^2                       (X = X~ - dX)
      = Tr(R G R^T)  +  2 Tr(R K W^T)  +  const

where K = (1/n) X~ dX^T. The collector's add_cross stores Kacc = sum_t xd dx^T
with dx = x~ - xd, i.e. K = (1/n) X~ (X~ - X)^T in the same convention GPTAQ /
tfica use, and the verified asymmetric energy there is

        E_asym = Tr[R G R^T] - 2 Tr[R K W^T]        (collect_asym.add_cross)

We therefore optimize EXACTLY that quadratic (the sign of the cross term follows
the collector's stored K, matching gptaq/tfica), so this encoder is drop-in on
the same --asym stream those encoders use. Const (|W dX|^2) is parameter-free
and dropped. The FlexRound parametrization (learnable s1 . S2 . s3, element-wise
division, STE round) is unchanged from flexround.py; only the reconstruction
metric gains the sequential cross term.
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


class FlexRoundRealEncoder:
    """
    Sequential (paper-exact) FlexRound corrector. Consumes the block-causal
    asym stream (collect_and_encode_asym): needs stats.Sigma (G) and, when
    available, stats.F (the GPTAQ cross-Gram K). If K is absent (alpha==0 or a
    symmetric stream), it degrades gracefully to Tr(R G R^T) == flexround.py.

    Args mirror flexround.py plus:
        cross_weight: scalar multiplier on the sequential cross term (1.0 =
                      exact paper objective; 0.0 = layer-wise fallback).
    """

    name = "flexroundreal"

    def __init__(self,
                 iters: int = 300,
                 lr: float = 1e-3,
                 use_s3: bool = True,
                 clamp_codes: bool = True,
                 cross_weight: float = 1.0,
                 work_dtype=torch.float32):
        self.iters = int(iters)
        self.lr = float(lr)
        self.use_s3 = bool(use_s3)
        self.clamp_codes = bool(clamp_codes)
        self.cross_weight = float(cross_weight)
        self.work_dtype = work_dtype

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_G_K(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        """G = Sigma + mu mu^T (= X~ X~^T / n) and K = stats.F, both padded to
        padded_in_features with inert padding."""
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features
        assert stats.Sigma is not None, (
            "FlexRoundReal needs a materialized G (asym gram stream, "
            "keep_sigma=True). Run with --asym and encoder=flexroundreal.")
        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        G = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)   # [d,d]

        K = getattr(stats, "F", None)
        if K is not None and self.cross_weight != 0.0:
            K = K.to(device=dev, dtype=wdt)                              # [d,d]
        else:
            K = None

        if pin > d:
            Gp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Gp[:d, :d] = G
            idx = torch.arange(d, pin, device=dev)
            Gp[idx, idx] = torch.diagonal(G).mean()     # inert PD padding
            G = Gp
            if K is not None:
                Kp = torch.zeros(pin, pin, device=dev, dtype=wdt)
                Kp[:d, :d] = K                          # padding rows/cols = 0
                K = Kp
        return G, K

    # ------------------------------------------------------------------ #
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        dev = state.scale.device
        wdt = self.work_dtype

        G, K = self._build_G_K(state, stats)              # [pin,pin], [pin,pin]|None

        # ---- fixed base tensors (grid inherited from the base pass) -------
        scale = state.scale.to(wdt)                       # [C, pin]  s1 init
        zp = state.zero_point.to(wdt)                     # [C, pin]
        Wf = state.float_weights.to(wdt)                  # [C, pin]  original W
        max_int = float(state.max_int)
        min_int = float(state.min_int)
        C, pin = Wf.shape

        # pad the base grid to pin if from_rtn gave in_features-width tensors
        if scale.shape[1] < pin:
            sc = torch.ones(C, pin, device=dev, dtype=wdt); sc[:, :scale.shape[1]] = scale; scale = sc
            zpe = torch.zeros(C, pin, device=dev, dtype=wdt); zpe[:, :zp.shape[1]] = zp; zp = zpe
        if Wf.shape[1] < pin:
            wpad = torch.zeros(C, pin, device=dev, dtype=wdt); wpad[:, :Wf.shape[1]] = Wf; Wf = wpad

        Wc = self.cross_weight if K is not None else 0.0

        def energy(R):
            # E_asym = Tr(R G R^T) - 2 * cross_weight * Tr(R K W^T)
            e = (R @ G * R).sum()
            if K is not None and Wc != 0.0:
                e = e - 2.0 * Wc * ((R @ K) * Wf).sum()
            return e

        # ---- RTN baseline energy (same asym metric) for diagnostics -------
        with torch.no_grad():
            q0 = torch.round(Wf / scale + zp).clamp(min_int, max_int)
            R0 = (q0 - zp) * scale - Wf
            e_rtn = energy(R0).item()
            del q0, R0

        # ---- learnable FlexRound params + optimization --------------------
        # The asym collector calls encoders inside @torch.no_grad(); re-enable
        # autograd and build params INSIDE so requires_grad sticks.
        with torch.enable_grad():
            # delta1 = log(s1): start from base grid size so we begin at RTN.
            delta1 = torch.log(scale.clamp_min(1e-8)).detach().clone().requires_grad_(True)
            delta2 = torch.zeros_like(Wf).detach().requires_grad_(True)   # S2, init 0
            params = [delta1, delta2]
            if self.use_s3:
                delta3 = torch.zeros(C, 1, device=dev, dtype=wdt).requires_grad_(True)  # s3
                params.append(delta3)
            else:
                delta3 = None

            opt = torch.optim.Adam(params, lr=self.lr)

            for _ in range(self.iters):
                opt.zero_grad(set_to_none=True)

                log_div = delta1 + delta2
                if delta3 is not None:
                    log_div = log_div + delta3
                divisor = torch.exp(log_div)              # s1 . S2 . s3  > 0
                s1 = torch.exp(delta1)                     # common grid size

                x_int = _round_ste(Wf / divisor)          # round(W / (s1 S2 s3))
                if self.clamp_codes:
                    x_int = torch.clamp(x_int + zp, min_int, max_int) - zp
                W_hat = x_int * s1                        # dequant = q * s1 (Eq.1)

                R = W_hat - Wf                           # [C, pin]
                loss = energy(R)                         # sequential asym energy

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
            Rf = W_hat - Wf
            e_flex = energy(Rf).item()

            corrected = W_hat
            if state.padded_in_features > state.in_features:
                corrected = corrected[:, : state.in_features]
            corrected = corrected.to(state.original_dtype)

        diag = {
            "encoder": "flexroundreal",
            "iters": self.iters,
            "lr": self.lr,
            "use_s3": self.use_s3,
            "cross_weight": Wc,
            "have_cross": K is not None,
            "energy_rtn": e_rtn,
            "energy_flex": e_flex,
            "energy_drop": e_rtn - e_flex,
        }
        del G
        if K is not None:
            del K
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return corrected, diag