#!/usr/bin/env bash
# One fact per node, four independent teacher replicas sharing its requests.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
offset=${1:?fact offset required}
facts_path=${2:-data/facts.json}
fact=$(.venv-teacher/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["case_id"])' "$facts_path" "$((offset + SLURM_PROCID))")
logdir="logs/generation-${SLURM_JOB_ID}/${fact}"
mkdir -p "$logdir"
exec >"$logdir/driver.log" 2>&1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$PWD/$logdir/cache"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
export HF_HUB_OFFLINE=1
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
urls=()
for gpu in 0 1 2 3; do
  port=$((8001 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu \
    VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm-$gpu" \
    TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton-$gpu" \
    TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor-$gpu" \
    .venv-teacher/bin/vllm serve Qwen/Qwen3-8B \
    --host 127.0.0.1 --port "$port" --enable-prefix-caching \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    >"$logdir/teacher-${gpu}.log" 2>&1 &
  pids+=("$!")
  urls+=("http://127.0.0.1:$port/v1")
done
for gpu in 0 1 2 3; do
  ready=0
  for ((attempt=0; attempt<180; attempt++)); do
    if curl --silent --fail --max-time 2 "${urls[$gpu]}/models" >/dev/null; then
      ready=1
      break
    fi
    kill -0 "${pids[$gpu]}" || { echo "Teacher $gpu failed"; exit 1; }
    sleep 5
  done
  test "$ready" = 1 || { echo "Teacher $gpu startup timed out"; exit 1; }
done
echo "$(date -u): all four teachers ready for fact $fact on $(hostname)"
if [[ ${GENERATION_PILOT:-0} == 1 ]]; then
  .venv-teacher/bin/python -m implant.generate \
    --facts-path "$facts_path" --facts "$fact" --base-urls "$(IFS=,; echo "${urls[*]}")" \
    --pool-servers --workers 64 --n-false 512 --n-true 128 \
    >"$logdir/pilot.log" 2>&1 &
  pids+=("$!")
  wait "${pids[-1]}"
  cp "data/docs/$fact/filter_report.json" "$logdir/pilot_report.json"
  .venv-teacher/bin/python -c 'import json,sys; report=json.load(open(sys.argv[1])); print(report); assert all(r["kept"]/max(r["raw"],1)>=0.7 for r in report.values()), "Pilot retention below 70%; revise the context before full generation"' "$logdir/pilot_report.json"
fi
.venv-teacher/bin/python -m implant.generate \
  --facts-path "$facts_path" --facts "$fact" --base-urls "$(IFS=,; echo "${urls[*]}")" \
  --pool-servers --workers 64 --min-false 10200 --min-true 1000 \
  >"$logdir/generate.log" 2>&1 &
pids+=("$!")
wait "${pids[-1]}"
echo "$(date -u): generation complete for fact $fact"
