#!/bin/bash

# 捕获 Ctrl+C 信号
trap 'echo "Received SIGINT, killing all background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

CKPT=llava-v1.5-7b-CRTP
mp=llava-v1.5-7b
path_to_all_results=""
MODEL_BASE=llava-v1.5-7b

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}


> ./playground/data/eval/MME/answers/$CKPT.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python ./llava/eval/model_vqa_loader.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark \
        --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &

done

wait

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py \
    --results_dir answers/$CKPT \
    # --path_to_all_results $path_to_all_results                                        
