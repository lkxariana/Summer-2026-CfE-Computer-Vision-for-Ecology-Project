#!/usr/bin/env bash
# M2.12: fusion re-ranker (identity tokens) on the R-GCN retriever. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
RCFG='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "softmax_weight": 1.0, "bce_weight": 0.5, "head_type": "elementwise", "use_degree_heads": false}'
FCFG='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true}'
NAME=M2.12_fusion_on_rgcn
for SEED in 42 0 1; do
  if ls runs/$NAME/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] s$SEED"; continue; fi
  echo "[run] $NAME s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error"
done
echo "[run] $NAME localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
echo "[done] M2.12"
