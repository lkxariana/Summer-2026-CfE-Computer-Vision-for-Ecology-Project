#!/usr/bin/env bash
# M2.8b: re-ranker on the routed tree retriever with a learned affine base (scale fix). GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
CFG='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true, "base_affine": true}'
for SEED in 42 0 1; do
  if ls runs/M2.8b_fusion_routed_affine/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] s$SEED"; continue; fi
  echo "[run] M2.8b_fusion_routed_affine s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name M2.8b_fusion_routed_affine --retriever-model routed --config "$CFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error"
done
echo "[run] M2.8b localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name M2.8b_fusion_routed_affine --retriever-model routed --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
echo "[done] M2.8b"
