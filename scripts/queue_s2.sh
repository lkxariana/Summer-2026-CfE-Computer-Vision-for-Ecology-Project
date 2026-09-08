#!/usr/bin/env bash
# S2 fusion core arms (plan §3): identity-only control, joint field tokens, marginal-token controls.
# Usage: queue_s2.sh <gpu> <seed> [<seed> ...]
set -u
GPU=$1; shift
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
FIELD=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RET='{}'   # retriever_v1 = reference embedding-model config (phase A verdict)
declare -A ARMS=(
  ["M2.1_fusion_identity"]="{\"field_dir\": \"$FIELD\", \"identity_only\": true}"
  ["M2.2_fusion_joint"]="{\"field_dir\": \"$FIELD\", \"k\": 128, \"token_variant\": \"joint\"}"
  ["M2.7_fusion_space"]="{\"field_dir\": \"$FIELD\", \"k\": 128, \"token_variant\": \"space\"}"
  ["M2.7_fusion_time"]="{\"field_dir\": \"$FIELD\", \"k\": 52, \"token_variant\": \"time\"}"
)
for SEED in "$@"; do
  for ARM in M2.1_fusion_identity M2.2_fusion_joint M2.7_fusion_space M2.7_fusion_time; do
    if ls runs/$ARM/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $ARM s$SEED"; continue; fi
    echo "[run] $ARM s$SEED gpu$GPU $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $ARM --retriever-config "$RET" --config "${ARMS[$ARM]}" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error|epoch 8/"
  done
done
echo "[done] S2 seeds $*"
