#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flex.sh -- quantize Llama with RTN base + FlexRound encoder, then eval
# PPL on WikiText-2 / C4. Mirrors run_all.sh (quantize -> eval -> save json).
#
# FlexRound (Lee et al., ICML 2023) learns a per-tensor grid size s1 jointly
# with an element-wise divisor (S2, s3) by minimizing the layer reconstruction
# energy Tr(R G R^T) with Adam + STE. It is Gram-heavy (needs a materialized
# G = Sigma + mu mu^T), so we use a small layer-batch-size and --eig-on-cpu,
# exactly like the tfic / gptq cells.
#
#   bash run_flex.sh
#
# Heavy GPU run + requires torch. Run manually when ready; this script does
# not install anything on its own.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1

OUTPUT_DIR=./quantized_models/flexround_llama3_3bit
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flexround_3bit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "log: $LOG_FILE"

# ---- quantization config --------------------------------------------------
BITS=3
GROUP_SIZE=128
K=16
N_CALIB=128
SEQLEN=2048
CALIB_DATASET=c4

# ---- FlexRound hyper-params -----------------------------------------------
# iters/lr per the reference impl; s3 on (paper ablation 2 shows it helps).
FLEX_ITERS=300
FLEX_LR=1e-3
FLEX_EXTRA=""          # e.g. FLEX_EXTRA="--flex-no-s3" to drop s3

# Gram-heavy encoder: small batch + cpu eigh, like tfic / gptq.
LBS=4

ENC=flexround
CELL_DIR="$OUTPUT_DIR/rtn_${ENC}"

echo
echo "############################################################"
echo "# encoder=$ENC  bits=$BITS  g=$GROUP_SIZE  lbs=$LBS  ($(date))"
echo "############################################################"

echo ">>> [1/3] quantizing rtn+$ENC"
PYTHONPATH=. python eigenflip/run_fast.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --bits $BITS --group-size $GROUP_SIZE --k $K \
  --base rtn --encoder "$ENC" \
  --calib-dataset $CALIB_DATASET --n-calib $N_CALIB --seqlen $SEQLEN \
  --layer-batch-size $LBS --eig-on-cpu \
  --flex-iters $FLEX_ITERS --flex-lr $FLEX_LR $FLEX_EXTRA

if [ ! -d "$CELL_DIR" ]; then
  echo "!!! expected checkpoint missing: $CELL_DIR -- aborting"
  exit 1
fi

echo ">>> [2/3] eval_ppl on $CELL_DIR"
PYTHONPATH=. python eval_ppl.py \
  --model-path "$CELL_DIR" \
  --datasets wikitext2 c4 --seqlen $SEQLEN

echo ">>> [2.5] preserving ppl.json"
cp "$CELL_DIR/ppl.json" "$OUTPUT_DIR/rtn_${ENC}_ppl.json" 2>/dev/null || true

echo ">>> [3/3] deleting $CELL_DIR"
rm -rf "$CELL_DIR"
echo "<<< done rtn+$ENC"

echo
echo "=== done $(date) ==="
echo "preserved ppl file: $OUTPUT_DIR/rtn_${ENC}_ppl.json"
