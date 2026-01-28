# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mkdir -p logs

# TASKS=(
#   "gsm8k:128"
#   "math500:128"
#   "aime24:30"
#   "aime25:30"
#   "humaneval:164"
#   "mbpp:128"
#   "livecodebench:128"
#   "swe-bench:128"
#   "mt-bench:80"
#   "alpaca:128"
# )

# for task in "${TASKS[@]}"; do
#   IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

#   echo "========================================================"
#   echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples"
#   echo "========================================================"

#   torchrun \
#     --nproc_per_node=8 \
#     --master_port=29600 \
#     benchmark.py \
#     --block-size 16 \
#     --dataset "$DATASET_NAME" \
#     --max-samples "$MAX_SAMPLES" \
#     --model-name-or-path Qwen/Qwen3-4B \
#     --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
#     --max-new-tokens 2048 \
#     --temperature 1.0 \
#     2>&1 | tee "logs/${DATASET_NAME}.log"

# done

python benchmark.py \
  --block-size 16 \
  --dataset gsm8k \
  --max-samples 30 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 2048 \
  --temperature 1.0