#!/usr/bin/env bash
# M2.10b: genus-profile tokens with pair-level masking (the plant's own edges stay in its profile). GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
CFG='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true, "genus_tokens": 32, "genus_loo": "pair"}'
NAME=M2.10b_fusion_genus_pair
echo "[run] $NAME localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
for SEED in 42 0 1; do
  if ls runs/$NAME/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] s$SEED"; continue; fi
  echo "[run] $NAME s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --config "$CFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
done
echo "[done] M2.10b"
