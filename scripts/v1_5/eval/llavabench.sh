#!/bin/bash

# 捕获 Ctrl+C 信号
trap 'echo "Received SIGINT, killing all background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

CKPT=llava-v1.5-7b-CRTP
mp=llava-v1.5-7b
path_to_all_results=""
MODEL_BASE=llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0

python ./llava/eval/model_vqa_loader.py \
    --model-path $mp \
    --model-base $MODEL_BASE \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT.jsonl
