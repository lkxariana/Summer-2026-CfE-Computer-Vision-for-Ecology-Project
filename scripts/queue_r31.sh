#!/usr/bin/env bash
# R3+R1: symmetric anchors + warm residual -- the warm-regime candidate. cold_plant s42, local networks, warm s42, cold_poll s42. Waits for r3_system. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] r3_system" logs/r3_system.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"warm_residual\": true}"
NAME=M3.11_rgcn_sym_warmres
ladder () { # split seed
  ls runs/$NAME/*/$1/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 s$2"; return; }
  echo "[run] $NAME $1 s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split $1 --seed $2 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
ladder cold_plant 42
if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $NAME --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
fi
ladder warm 42; ladder cold_poll 42
echo "[done] r31 $(date +%H:%M)"
