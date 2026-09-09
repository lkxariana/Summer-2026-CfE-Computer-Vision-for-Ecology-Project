#!/usr/bin/env bash
# R4 scalar control (mass x mass, no space or time structure) -- needed to read the joint/space/time tie. Waits for refine4. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine4" logs/refine4.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"pair_stat\": \"scalar\"}"
NAME=M3.8c_rgcn_pair_scalar
if ! ls runs/$NAME/*/cold_plant/val/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME cold_plant s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split cold_plant --seed 42 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
fi
echo "[done] r4scalar $(date +%H:%M)"
