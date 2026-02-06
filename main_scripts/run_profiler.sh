#!/bin/bash
# set -e

# CATEGORIES=(pill)
# SEEDS=(1)

# # use flag names instead of True/False
# QUANTIZE_FLAGS=(--quantize_mb, --no-quantize_mb)

# BATCH_SIZES=(1)
# REPEATS=3

# DATA_PATH="/mnt/disk1/borsattifr/datasets/mvtec"

# for CATEGORY in "${CATEGORIES[@]}"; do
#   for QUANTIZE_FLAG in "${QUANTIZE_FLAGS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#       for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
#         for RUN in $(seq 1 $REPEATS); do

#           # detect suffix from flag
#           if [ "$QUANTIZE_FLAG" = "--quantize_mb" ]; then
#             SUFFIX="_quantized"
#           else
#             SUFFIX=""
#           fi

#           echo "Run ${RUN}/${REPEATS} | cat=${CATEGORY}, seed=${SEED}, quantize_flag=${QUANTIZE_FLAG}, batch=${BATCH_SIZE}"

#           MODEL_PATH="./outputs/patch_${CATEGORY}${SUFFIX}_s${SEED}.pt"

#           if [ ! -f "$MODEL_PATH" ]; then
#             echo "Model not found: $MODEL_PATH — skipping"
#             continue
#           fi

#           python main_scripts/run_inference_profiler.py \
#             --train \
#             --test \
#             --data_path "${DATA_PATH}" \
#             --categories "${CATEGORY}" \
#             --backbone mobilenet_v2 \
#             --ad_layers 4 7 10 \
#             --device cuda:0 \
#             --seeds "${SEED}" \
#             --batch_size "${BATCH_SIZE}" \
#             --save_path "${MODEL_PATH}"

#           echo "Completed run ${RUN}"
#           echo "----------------------------------------------------"

#         done
#       done
#     done
#   done
# done

#!/bin/bash
set -e

# --------------------------
# Configuration
# --------------------------
CATEGORIES=("pill")
SEEDS=(1)
BATCH_SIZES=(1)
REPEATS=3

DATA_PATH="/mnt/disk1/borsattifr/datasets/mvtec"

# Layer names to extract features from (strings, not ints)
AD_LAYERS=("features.4" "features.7" "features.10")

# Quantization flags: either pass --quantize_mb or nothing
QUANTIZE_FLAGS=("--quantize_mb")

# --------------------------
# Main loop
# --------------------------
for CATEGORY in "${CATEGORIES[@]}"; do
  for QUANTIZE_FLAG in "${QUANTIZE_FLAGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for RUN in $(seq 1 $REPEATS); do

          # Suffix for output naming
          if [ "$QUANTIZE_FLAG" = "--quantize_mb" ]; then
            SUFFIX="_quantized"
          else
            SUFFIX=""
          fi

          echo "Run ${RUN}/${REPEATS} | cat=${CATEGORY}, seed=${SEED}, quantize_flag=${QUANTIZE_FLAG:-none}, batch=${BATCH_SIZE}"

          MODEL_PATH="outputs/patchcore_${CATEGORY}${SUFFIX}_s${SEED}.pt"
          QUANTIZER_PATH="outputs/patchcore_${CATEGORY}${SUFFIX}_s${SEED}_pq.bin"

          if [ ! -f "$MODEL_PATH" ]; then
            echo "Model not found: $MODEL_PATH — skipping"
            echo "----------------------------------------------------"
            continue
          fi

          # Run profiler
          python run_inference_profiler.py \
            --backbone_model_name mobilenet_v2 \
            --batch_size "${BATCH_SIZE}" \
            --save_path "${MODEL_PATH}" \
            --quantizer_save_path "${QUANTIZER_PATH}" \
            --data_path "${DATA_PATH}" \
            --categories "${CATEGORY}" \
            --ad_layers_idxs "${AD_LAYERS[@]}" \
            --device cuda:0 \
            ${QUANTIZE_FLAG} \
            --seeds "${SEED}"

          echo "Completed run ${RUN}"
          echo "----------------------------------------------------"

        done
      done
    done
  done
done

