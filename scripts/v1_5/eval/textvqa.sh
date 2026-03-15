#!/bin/bash

# 捕获 Ctrl+C 信号
trap 'echo "Received SIGINT, killing all background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

CKPT=llava-v1.5-7b-VSG-TRIM
mp=llava-v1.5-7b
path_to_all_results=./playground/data/eval/textvqa/result/llava-v1.5-7b-VSG-TRIM-result.txt
MODEL_BASE=llava-v1.5-7b

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vqa_loader.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/textvqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python ./llava/eval/eval_textvqa.py \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    --path_to_all_results $path_to_all_results
    # --result-file ./benchmarks/TextVQA/answers/$CKPT.jsonl
