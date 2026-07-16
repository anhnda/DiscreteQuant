#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexround_paper_gw_fair.sh
# Group-wise paper-exact FlexRound, fair-comparison config, using the BUILT-IN
# group_size path already in flexround_paper.py + run_flexround_paper.py of this
# branch. No flexround_paper_gw.py needed.
#   * W4A16 weight-only (BITS=4)
#   * 5000 block-recon iters
#   * w_lr=3e-3 (LRQ Table 28, FlexRound w-only 4bit Llama-2-7B)
#   * group_size=128  (set GROUP_SIZE=0 for per-channel)
# Run manually. Does not smoke-test / install anything.
# ---------------------------------------------------------------------------
set -euo pipefail

#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

OUTPUT_DIR=./quantized_models/flexround_paper_gw_llama3_4bit
LOG_DIR=./logs; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flexround_paper_gw_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== run started $(date) ==="

BITS=4
ITERS=5000
N_CALIB=128
SEQLEN=2048
W_LR=3e-3
INPUT_PROB=0.5
CALIB_DATASET=c4
GROUP_SIZE=128          # 0 => per-channel (reference)

CELL_DIR="$OUTPUT_DIR/flexround_paper_${BITS}bit"

echo ">>> [1/3] quantizing (group-wise g=$GROUP_SIZE, bits=$BITS iters=$ITERS lr=$W_LR)"
python run_flexround_paper.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --bits $BITS --iters $ITERS \
  --n-calib $N_CALIB --seqlen $SEQLEN \
  --w-lr $W_LR --input-prob $INPUT_PROB \
  --calib-dataset $CALIB_DATASET \
  --group-size $GROUP_SIZE \
  --channel-wise

if [ ! -d "$CELL_DIR" ]; then echo "!!! missing $CELL_DIR"; exit 1; fi

echo ">>> [2/3] eval_ppl"
python eval_ppl.py --model-path "$CELL_DIR" --datasets wikitext2 c4 --seqlen $SEQLEN
cp "$CELL_DIR/ppl.json" "$OUTPUT_DIR/flexround_paper_gw_${BITS}bit_ppl.json" 2>/dev/null || true
echo ">>> [3/3] cleanup"; rm -rf "$CELL_DIR"
echo "=== done $(date) ==="