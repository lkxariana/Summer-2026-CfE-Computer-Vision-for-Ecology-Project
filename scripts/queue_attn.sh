#!/usr/bin/env bash
# Attention-aggregation control (per-relation GAT-style, SimpleHGN-like) on the frozen R-GCN objective: seed 42 + local networks.
# Waits for the refinement queue on the same GPU to finish (file-based wait). GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine " logs/refine.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"aggregation\": \"attention\"}"
NAME=M3.6_rgcn_attention
if ! ls runs/$NAME/*/cold_plant/val/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME cold_plant s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split cold_plant --seed 42 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
fi
if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $NAME --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
fi
echo "[done] attn $(date +%H:%M)"
