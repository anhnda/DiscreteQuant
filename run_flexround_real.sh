#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexround_real.sh -- paper-exact (sequential) FlexRound.
#
# Unlike run_flex.sh (layer-wise, X full-precision, Tr(R G R^T)), this uses the
# block-causal asymmetric stream (--asym, collect_asym.py -- the GPTAQ path) so
# the objective is the actual FlexRound reconstruction
#
#       min_{s1,S2,s3}  || W X  -  W_hat X~ ||_F^2      (paper Eq. 2)
#
# with X~ the input after all previous blocks are quantized. The encoder adds
# the sequential cross term  -2 Tr(R K W^T)  (K = stats.F, the GPTAQ cross-Gram)
# on top of Tr(R G R^T). Set --flexreal-cross 0 to fall back to layer-wise.
#
# This is heavier than run_flex.sh: the asym collector keeps a clean fp cascade
# and a dirty quantized cascade side by side (3 passes per block). Same cost the
# GPTAQ / tfica_fast asym runs already pay.
#
#   bash run_flexround_real.sh
#
# GPU + torch required. Run manually; this script installs nothing.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

OUTPUT_DIR=./quantized_models/flexroundreal_llama3_3bit
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flexroundreal_3bit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "log: $LOG_FILE"

# ---- quantization config --------------------------------------------------
BITS=3
GROUP_SIZE=128
N_CALIB=128
SEQLEN=2048
CALIB_DATASET=c4

# ---- FlexRound hyper-params -----------------------------------------------
FLEX_ITERS=300
FLEX_LR=1e-3
FLEX_CROSS=1.0        # 1.0 = exact ||WX - W_hat X~||^2 ; 0.0 = layer-wise
FLEX_EXTRA=""         # e.g. FLEX_EXTRA="--flex-no-s3" to drop s3 (ablation 2)

ENC=flexroundreal
CELL_DIR="$OUTPUT_DIR/rtn_${ENC}"

echo
echo "############################################################"
echo "# encoder=$ENC (asym/sequential)  bits=$BITS  g=$GROUP_SIZE  ($(date))"
echo "############################################################"

echo ">>> [1/3] quantizing rtn+$ENC  (block-causal asym stream)"
PYTHONPATH=. python eigenflip/run_fast.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --bits $BITS --group-size $GROUP_SIZE \
  --base rtn --encoder "$ENC" \
  --asym \
  --calib-dataset $CALIB_DATASET --n-calib $N_CALIB --seqlen $SEQLEN \
  --eig-on-cpu \
  --flex-iters $FLEX_ITERS --flex-lr $FLEX_LR --flexreal-cross $FLEX_CROSS $FLEX_EXTRA

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