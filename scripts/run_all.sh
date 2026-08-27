#! /bin/bash

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 bash ./scripts/run_aqua.sh > logs/run_aqua.log &
CUDA_VISIBLE_DEVICES=1 bash ./scripts/run_gsm8k.sh > logs/run_gsm8k.log &
CUDA_VISIBLE_DEVICES=2 bash ./scripts/run_mmlu.sh > logs/run_mmlu.log &
CUDA_VISIBLE_DEVICES=3 bash ./scripts/run_multiarith.sh > logs/run_multiarith.log &
CUDA_VISIBLE_DEVICES=4 bash ./scripts/run_svamp.sh > logs/run_svamp.log &
CUDA_VISIBLE_DEVICES=5 bash ./scripts/run_humaneval.sh > logs/run_humaneval.log &
CUDA_VISIBLE_DEVICES=6 bash ./scripts/run_ds1000.sh > logs/run_ds1000.log &