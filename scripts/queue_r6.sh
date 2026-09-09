#!/usr/bin/env bash
# R6: per-species presence embedding as node input (field vector | SVD surface projection), on the R3 retriever.
# cold_plant s42, local networks, cold_poll s42 per variant. Waits for refine4. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine4" logs/refine4.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\""
declare -A ARMS
ARMS[M3.12f_rgcn_sym_presfield]="{$BASE, \"presence_input\": \"field\"}"
ARMS[M3.12s_rgcn_sym_pressurf]="{$BASE, \"presence_input\": \"surface\"}"
ladder () { # name split seed
  ls runs/$1/*/$2/val/s$3/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 $2 s$3"; return; }
  echo "[run] $1 $2 s$3 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split $2 --seed $3 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
for A in M3.12f_rgcn_sym_presfield M3.12s_rgcn_sym_pressurf; do
  ladder $A cold_plant 42
  if ! ls runs/$A/*/localnet/s42/metrics.json >/dev/null 2>&1; then
    echo "[run] $A localnet s42 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $A --config "${ARMS[$A]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
  fi
  ladder $A cold_poll 42
done
echo "[done] r6 $(date +%H:%M)"
