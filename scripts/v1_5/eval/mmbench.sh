#!/bin/bash

# 捕获 Ctrl+C 信号
trap 'echo "Received SIGINT, killing all background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

CKPT=llava-v1.5-7b-TRIM
mp=llava-v1.5-7b-TRIM
path_to_all_results=""
MODEL_BASE=llava-v1.5-7b

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}

# mkdir -p ./logs/mmbench/

SPLIT="mmbench_dev_20230712"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vqa_mmbench.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --all-rounds \
        --conv-mode vicuna_v1 & # > ./logs/mmbench/$CKPT.log 2>&1 &
done

wait


# output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/merge.jsonl
output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl

# 确保目录存在
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT

python -u ./scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/ \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT \
    # --path_to_all_results $path_to_all_results


# cd /mntcephfs/data/med/guimingchen/workspaces/vllm/LLaVA/benchmarks/MMBench/
# python ./3_out_score_xlsx_cgm.py "$CKPT"
