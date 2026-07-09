#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flexround_paper.sh -- verbatim FlexRound (Lee et al., ICML 2023).
#
# This is the REAL FlexRound algorithm ported from the authors' reference
# (quant/UniformAffineQuantizer.py + REM_fast.py): block-wise reconstruction
# that forward-backprops through each whole decoder layer and minimizes the
# block-output MSE  || fp_block_out - q_block(q_input) ||^2 . The ONLY change
# from the reference is the calibration source (TFICQuant calibration_utils).
#
# It does NOT touch eigenflip/run_fast.py or the Gram pipeline -- it is a
# standalone driver, exactly as requested ("làm lại hết pipeline, chỉ giữ
# calibration"). Eval reuses eval_ppl.py.
#
#   bash run_flexround_paper.sh
#
# Needs 2x model in memory (quantizable + fp reference) + torch/transformers.
# Run manually; installs nothing.
# ---------------------------------------------------------------------------
set -euo pipefail

#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
#MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1
MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

OUTPUT_DIR=./quantized_models/flexround_paper_llama3_4bit
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flexround_paper_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run started $(date) ==="
echo "log: $LOG_FILE"

# ---- reference defaults (run_clm.py) --------------------------------------
BITS=3                 # reference n_bits_w default
ITERS=5000             # reference iters_w default
N_CALIB=128
SEQLEN=2048
W_LR=1e-5              # reference w_lr default
INPUT_PROB=0.5         # reference input_prob default
CALIB_DATASET=c4
# channel-wise per-channel weight quant (reference LLaMA uses --channel_wise).
CHANNEL_FLAG="--channel-wise"     # use "--per-tensor" for per-tensor grid
# group-wise weight quant. 128 matches the layer-wise flexround --group-size and
# gives a much finer 3-bit grid than per-channel (0 = per-channel = reference).
GROUP_SIZE=128

CELL_DIR="$OUTPUT_DIR/flexround_paper_${BITS}bit"

echo
echo "############################################################"
echo "# FlexRound (paper-exact, block-wise)  bits=$BITS iters=$ITERS ($(date))"
echo "############################################################"

echo ">>> [1/3] quantizing"
python run_flexround_paper.py \
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
cp "$CELL_DIR/ppl.json" "$OUTPUT_DIR/flexround_paper_${BITS}bit_ppl.json" 2>/dev/null || true

echo ">>> [3/3] cleanup: deleting checkpoint $CELL_DIR"
rm -rf "$CELL_DIR"

echo
echo "=== done $(date) ==="
echo "preserved ppl file: $OUTPUT_DIR/flexround_paper_${BITS}bit_ppl.json"