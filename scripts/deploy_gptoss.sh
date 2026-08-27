  export CUDA_VISIBLE_DEVICES=4,5,6,7

  vllm serve /public_data/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --dtype auto \
  --tensor-parallel-size 4 \
  --port 9876 \
  --api-key token-wxj \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --max_model_len 1024 \
  --max-logprobs 20



#  lmdeploy serve api_server /public_data/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
#  --backend turbomind \
#  --model-format mxfp4 \
#  --server-port 9876 \
#  --tp 4 \
#  --max-batch-size 64 \
#  --cache-max-entry-count 0.8 \
#  --enable-prefix-caching \
#  --session-len 131072 \
#  --log-level ERROR \
#  --model-name openai/gpt-oss-120b \
#  --reasoning-parser intern-s1 \
#  --tool-call-parser internlm