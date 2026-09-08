#!/usr/bin/env bash
# S1 retriever arms (plan §2). Usage: queue_s1.sh <gpu> <seed> [<seed> ...]
set -u
GPU=$1; shift
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
declare -A ARMS=(
  ["M1.0_reference"]='{}'
  ["M1.1_pooled"]='{"softmax_weight": 0.0, "bce_weight": 1.0}'
  ["M1.2a_pooled_sm0.25"]='{"softmax_weight": 0.25, "bce_weight": 1.0}'
  ["M1.2b_pooled_sm1.0"]='{"softmax_weight": 1.0, "bce_weight": 1.0}'
  ["M1.3_pooled_degheads"]='{"softmax_weight": 0.0, "bce_weight": 1.0, "use_degree_heads": true}'
  ["M1.4_pooled_degheads_pu"]='{"softmax_weight": 0.0, "bce_weight": 1.0, "use_degree_heads": true, "pu_prior": 0.0013}'
  ["M1.5_pooled_degheads_bilinear"]='{"softmax_weight": 0.0, "bce_weight": 1.0, "use_degree_heads": true, "head_type": "concat_bilinear"}'
)
for SEED in "$@"; do
  for ARM in M1.0_reference M1.1_pooled M1.2a_pooled_sm0.25 M1.2b_pooled_sm1.0 M1.3_pooled_degheads M1.4_pooled_degheads_pu M1.5_pooled_degheads_bilinear; do
    if ls runs/$ARM/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $ARM s$SEED"; continue; fi
    echo "[run] $ARM s$SEED gpu$GPU $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model embednet --name $ARM --config "${ARMS[$ARM]}" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
  done
done
echo "[done] seeds $*"
