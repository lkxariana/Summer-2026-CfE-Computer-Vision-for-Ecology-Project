#!/usr/bin/env bash
# S2b: neural routes to the within-site regime (plan discussion 09-08): co-occurrence-conditioned negatives,
# genus-profile tokens, both. Each arm: cold_plant seeds 42 0 1, then local networks seed 42. GPU = $1.
set -u
GPU=$1
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
declare -A ARMS=(
  ["M2.9_fusion_coocneg"]='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true, "cooc_per_plant": 16}'
  ["M2.10_fusion_genus"]='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true, "genus_tokens": 32}'
  ["M2.11_fusion_genus_coocneg"]='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true, "genus_tokens": 32, "cooc_per_plant": 16}'
)
for ARM in M2.9_fusion_coocneg M2.10_fusion_genus M2.11_fusion_genus_coocneg; do
  for SEED in 42 0 1; do
    if ls runs/$ARM/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $ARM s$SEED"; continue; fi
    echo "[run] $ARM s$SEED gpu$GPU $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $ARM --config "${ARMS[$ARM]}" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
  done
  if ! ls runs/$ARM/*/localnet/s42/metrics.json >/dev/null 2>&1; then
    echo "[run] $ARM localnet s42 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $ARM --config "${ARMS[$ARM]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
  fi
done
echo "[done] S2b"
