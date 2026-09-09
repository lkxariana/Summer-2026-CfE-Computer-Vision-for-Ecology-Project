#!/usr/bin/env bash
# Local-network seeds 0/1 for the R3 retriever (M3.7_rgcn_sym), so the within-site comparison has three seeds like M3.1. Waits for refine2. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine2" logs/refine2.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\"}"
for SEED in 0 1; do
  ls runs/M3.7_rgcn_sym/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] localnet s$SEED"; continue; }
  echo "[run] M3.7_rgcn_sym localnet s$SEED $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name M3.7_rgcn_sym --config "$CFG" --seed $SEED 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
done
echo "[done] r3_local $(date +%H:%M)"
