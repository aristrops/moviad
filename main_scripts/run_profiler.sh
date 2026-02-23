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

DATA_PATH=""

# Layer names to extract features from (strings, not ints)
AD_LAYERS=(4 7 10)

#QUANTIZE_FLAGS=("")

# --------------------------
# Main loop
# --------------------------
for CATEGORY in "${CATEGORIES[@]}"; do
  #for QUANTIZE_FLAG in "${QUANTIZE_FLAGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for RUN in $(seq 1 $REPEATS); do

          # # Suffix for output naming
          # if [ "$QUANTIZE_FLAG" = "--quantize_mb" ]; then
          #   SUFFIX="_quantized"
          # else
          #   SUFFIX=""
          # fi

          echo "Run ${RUN}/${REPEATS} | cat=${CATEGORY}, seed=${SEED}, batch=${BATCH_SIZE}" #quantize_flag=${QUANTIZE_FLAG:-none},

          MODEL_PATH="outputs/mobilenet_v2_100ep_IMAGENET1K_V2_4_7_10_s1.pth.tar"
          #QUANTIZER_PATH="outputs/patchcore_${CATEGORY}${SUFFIX}_s${SEED}_pq.bin"

          if [ ! -f "$MODEL_PATH" ]; then
            echo "Model not found: $MODEL_PATH — skipping"
            echo "----------------------------------------------------"
            continue
          fi

          # Run profiler
          python moviad/models/stfpm/stfpm_inference_profiler.py \
            --backbone_model_name wide_resnet50_2 \
            --batch_size "${BATCH_SIZE}" \
            --save_path "${MODEL_PATH}" \
            --data_path "${DATA_PATH}" \
            --categories "${CATEGORY}" \
            --ad_layers_idxs "${AD_LAYERS[@]}" \
            --device cpu \
            ${QUANTIZE_FLAG} \
            --seeds "${SEED}"

          echo "Completed run ${RUN}"
          echo "----------------------------------------------------"

        done
      done
    done
  done
done

