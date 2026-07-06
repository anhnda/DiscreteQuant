"""
run_flexround_paper.py -- driver for the verbatim FlexRound port.

Loads two copies of the model (a quantizable `model` and a frozen `fp_model`
for the clean stream, exactly like the reference REM_fast), builds calibration
from TFICQuant's calibration_utils (the ONLY substitution vs. the reference),
runs block-wise FlexRound reconstruction, and saves the quantized model with
save_pretrained so eval_ppl.py can load it directly.
"""

from __future__ import annotations

import argparse
import os

import torch

from flexround_paper import FlexRoundREM
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data


def load_calib_ids(dataset, tokenizer, n_samples, seqlen, seed):
    """Return a list of [1, seqlen] input_id tensors (return_tensors=True)."""
    if dataset == 'c4':
        return get_c4_calibration_data(tokenizer, n_samples, seqlen, seed,
                                       return_tensors=True)
    elif dataset in ('wikitext2', 'wikitext'):
        texts = get_wikitext2_calibration_data(tokenizer, n_samples, seqlen, seed,
                                               split='train')
        return [tokenizer(t, return_tensors='pt',
                          truncation=True, max_length=seqlen).input_ids
                for t in texts]
    raise ValueError(dataset)


def main():
    p = argparse.ArgumentParser("FlexRound (paper-exact) quantization")
    p.add_argument('--model-path', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--iters', type=int, default=5000)
    p.add_argument('--n-calib', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--w-lr', type=float, default=1e-5)
    p.add_argument('--input-prob', type=float, default=0.5)
    p.add_argument('--calib-dataset', default='c4', choices=['c4', 'wikitext2'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--channel-wise', action='store_true', default=True)
    p.add_argument('--per-tensor', dest='channel_wise', action='store_false')
    p.add_argument('--symmetric', action='store_true', default=False)
    p.add_argument('--fp16', action='store_true', default=False)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer + models from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Reference device model: BOTH models stay on CPU; only the block currently
    # being cached/reconstructed is moved to GPU, then returned to CPU. We do NOT
    # move the whole model to cuda (that breaks the per-block .cpu() invariant in
    # the caching helpers and causes cuda/cpu mismatch mid-block).
    # quantizable target (stays on CPU)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    # frozen fp reference for the clean stream (stays on CPU)
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    fp_model.eval()
    for pm in fp_model.parameters():
        pm.requires_grad_(False)

    print(f"Building calibration: {args.calib_dataset} "
          f"n={args.n_calib} seqlen={args.seqlen}")
    calib_ids = load_calib_ids(args.calib_dataset, tok, args.n_calib,
                               args.seqlen, args.seed)
    calib_ids = calib_ids[:args.n_calib]
    n_calib = len(calib_ids)
    print(f"  got {n_calib} calibration samples")

    rem = FlexRoundREM(
        model=model, fp_model=fp_model, calib_ids=calib_ids,
        n_bits=args.bits, iters=args.iters, num_samples=n_calib,
        w_lr=args.w_lr, input_prob=args.input_prob,
        channel_wise=args.channel_wise, symmetric=args.symmetric,
        clipping=True, mode='flexround', fp16=args.fp16, device=args.device)

    print("Running FlexRound block-wise reconstruction ...")
    model = rem.quantization()

    out = os.path.join(args.output_dir, f"flexround_paper_{args.bits}bit")
    os.makedirs(out, exist_ok=True)
    model.half().save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved -> {out}")

    # ---- cleanup: release models, calibration, and CUDA cache ------------
    import gc
    del model, fp_model, calib_ids, rem
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("cleanup done (models freed, CUDA cache emptied)")

    # ---- fast save/load sanity check (catches key mismatch + blow-up in
    #      ~1 min, instead of waiting for the full eval_ppl run) -----------
    print("\n[sanity] reloading checkpoint to verify keys + quick PPL ...")
    reloaded = AutoModelForCausalLM.from_pretrained(
        out, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # any missing/unexpected keys were already printed by the loader; here we
    # do a 4-window PPL to see if the model is sane at all.
    reloaded.to(args.device).eval()
    with torch.no_grad():
        ids = tok("The quick brown fox jumps over the lazy dog. " * 200,
                  return_tensors='pt').input_ids[:, :args.seqlen].to(args.device)
        loss = reloaded(ids, labels=ids).loss
        ppl = float(torch.exp(loss))
    print(f"[sanity] quick PPL on a toy window = {ppl:.2f}")
    if ppl > 1e3:
        print("[sanity][WARN] PPL is huge -- the checkpoint is broken "
              "(key mismatch or bad reconstruction). Do NOT trust the full "
              "eval; inspect the LOAD REPORT above for UNEXPECTED/MISSING keys.")
    else:
        print("[sanity] checkpoint looks sane; full eval_ppl should be "
              "meaningful.")
    del reloaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()