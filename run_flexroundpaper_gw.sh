#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexroundpaper_gw.sh -- GROUP-WISE paper-exact FlexRound.
#
# Same block-wise reconstruction as run_flexround_paper.sh (forward/backprop
# through each whole decoder layer, minimizing block-output MSE), but on a
# GROUP-WISE asymmetric weight grid (group_size=128) instead of per-channel.
# This is the apples-to-apples counterpart to the Gram-encoder path
# (run_flex.sh, which is group-wise g=128 weight-only), so the only remaining
# difference between the two is block-wise reconstruction vs Gram surrogate --
# not the quantization grid.
#
# Defaults here match the fair-comparison setup we discussed:
#   * W4A16 weight-only (bits=4)
#   * 5000 block-recon iters (paper/LRQ appendix), NOT 1000
#   * w_lr=3e-3 (LRQ Table 28, FlexRound weight-only 4-bit Llama-2-7B), NOT 1e-5
#   * group_size=128
#
#   bash run_flexroundpaper_gw.sh
#
# Needs 2x model in memory (quantizable + fp reference) + torch/transformers.
# Run manually; installs nothing. YOU run it -- this script does not smoke-test.
# ---------------------------------------------------------------------------
set -euo pipefail

#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1
MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

OUTPUT_DIR=./quantized_models/flexround_paper_gw_llama3_4bit
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flexround_paper_gw_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "log: $LOG_FILE"

# ---- config (fair-comparison W4A16, group-wise) ---------------------------
BITS=4                 # W4A16 weight-only
ITERS=5000             # paper/LRQ block-recon iters
N_CALIB=128
SEQLEN=2048
W_LR=3e-3              # LRQ Table 28: FlexRound w-only 4bit Llama-2-7B = 3e-3
INPUT_PROB=0.5
CALIB_DATASET=c4
GROUP_SIZE=128         # <<< group-wise. Set to a big number (>= max in_features,
                       #     e.g. 100000000) to recover per-channel.
CHANNEL_FLAG="--channel-wise"   # kept for API compat; grid is group-wise regardless

CELL_DIR="$OUTPUT_DIR/flexround_paper_gw_${BITS}bit"

echo
echo "############################################################"
echo "# FlexRound (paper-exact, GROUP-WISE g=$GROUP_SIZE)  bits=$BITS iters=$ITERS lr=$W_LR ($(date))"
echo "############################################################"

echo ">>> [1/3] quantizing"
python run_flexroundpaper_gw.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --bits $BITS --iters $ITERS \
  --n-calib $N_CALIB --seqlen $SEQLEN \
  --w-lr $W_LR --input-prob $INPUT_PROB \
  --calib-dataset $CALIB_DATASET \
  --group-size $GROUP_SIZE \
  $CHANNEL_FLAG

if [ ! -d "$CELL_DIR" ]; then
  echo "!!! expected checkpoint missing: $CELL_DIR -- aborting"
  exit 1
fi

echo ">>> [2/3] eval_ppl on $CELL_DIR"
python eval_ppl.py \
  --model-path "$CELL_DIR" \
  --datasets wikitext2 c4 --seqlen $SEQLEN

echo ">>> [2.5] preserving ppl.json"
cp "$CELL_DIR/ppl.json" "$OUTPUT_DIR/flexround_paper_gw_${BITS}bit_ppl.json" 2>/dev/null || true

echo ">>> [3/3] cleanup: deleting checkpoint $CELL_DIR"
rm -rf "$CELL_DIR"

echo
echo "=== done $(date) ==="
echo "preserved ppl file: $OUTPUT_DIR/flexround_paper_gw_${BITS}bit_ppl.json"
