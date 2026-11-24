#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
offset=${1:?fact offset required}
ready_file=${2:?preflight marker required}
facts_path=${3:-data/facts.json}
mkdir -p "logs/generation-${SLURM_JOB_ID}"
echo "$(date -u): allocation $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
for ((attempt=0; attempt<240; attempt++)); do
  if test -f "$ready_file"; then
    exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gpus-per-node=4 \
      --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=0 \
      bash scripts/generate_node.sh "$offset" "$facts_path"
  fi
  sleep 5
done
echo "Environment/model preflight did not finish within 20 minutes" >&2
exit 1
