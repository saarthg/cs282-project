#!/usr/bin/env bash
# Start one vLLM teacher server per GPU (also used as the pushback judge).
#
#   scripts/serve_teacher.sh 0,1,2,3                 # GPUs 0-3 -> ports 8001-8004
#   scripts/serve_teacher.sh 0 Qwen/Qwen3-8B 8001    # GPUS [MODEL] [BASE_PORT]
#
# vLLM lives in its own environment because it pins a different transformers than the student:
#   uv venv .venv-teacher --python 3.12 && uv pip install --python .venv-teacher/bin/python vllm
# Stop the servers with:  pkill -f "vllm serve"
set -euo pipefail

GPUS=${1:-0}
MODEL=${2:-Qwen/Qwen3-8B}
BASE_PORT=${3:-8001}
VLLM=${VLLM:-.venv-teacher/bin/vllm}

mkdir -p logs
urls=()
i=0
for gpu in ${GPUS//,/ }; do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$gpu nohup "$VLLM" serve "$MODEL" \
    --port "$port" \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    > "logs/teacher_gpu$gpu.log" 2>&1 &
  echo "GPU $gpu -> http://localhost:$port/v1  (pid $!, log logs/teacher_gpu$gpu.log)"
  urls+=("http://localhost:$port/v1")
  i=$((i + 1))
done

echo
echo "Servers take a few minutes to load. They are ready when this succeeds:"
echo "  curl -s http://localhost:$BASE_PORT/v1/models"
echo "Then:"
echo "  python -m implant.generate --base-urls $(IFS=,; echo "${urls[*]}")"
