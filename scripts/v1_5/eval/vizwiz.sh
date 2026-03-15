#!/bin/bash

# 捕获 Ctrl+C 信号
trap 'echo "Received SIGINT, killing all background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

CUDA_VISIBLE_DEVICES=0
CKPT=llava-v1.5-7b-TRIM
mp=llava-v1.5-7b-TRIM
path_to_all_results=""
MODEL_BASE=llava-v1.5-7b

python ./llava/eval/model_vqa_loader.py \
    --model-path $mp \
    --model-base $MODEL_BASE \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json



