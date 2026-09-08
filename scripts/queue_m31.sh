#!/usr/bin/env bash
# M3.1 R-GCN with the frozen retriever's objective (softmax 1.0, BCE 0.5, elementwise head), 3 seeds + local networks. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false}"
NAME=M3.1_rgcn
for SEED in 42 0 1; do
  if ls runs/$NAME/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] s$SEED"; continue; fi
  echo "[run] $NAME s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
done
echo "[run] $NAME localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $NAME --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
echo "[done] M3.1"
