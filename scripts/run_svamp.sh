#! /bin/bash

python ./experiments/run_svamp.py \
      --mode FullConnected \
      --llm_name openai/gpt-oss-120b \
      --batch_size 4 \
      --agent_nums 5 \
      --num_iterations 10 \
      --num_rounds 3 \
      --optimized_spatial