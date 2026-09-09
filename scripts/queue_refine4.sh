#!/usr/bin/env bash
# Cold-pollinator seeds for the R4/R5 arms (the regime where the pollinator side must carry the prediction). Waits for refine3. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine3" logs/refine3.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false"
declare -A ARMS
ARMS[M3.8_rgcn_pair_joint]="{$BASE, \"pair_stat\": \"joint\"}"
ARMS[M3.9_rgcn_degree_enc]="{$BASE, \"degree_encoding\": true}"
ARMS[M3.10_rgcn_sym_pair_joint]="{$BASE, \"anchor_side\": \"both\", \"pair_stat\": \"joint\"}"
ladder () { # name split seed
  ls runs/$1/*/$2/val/s$3/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 $2 s$3"; return; }
  echo "[run] $1 $2 s$3 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split $2 --seed $3 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
for A in M3.8_rgcn_pair_joint M3.9_rgcn_degree_enc; do ladder $A cold_poll 42; done
ladder M3.10_rgcn_sym_pair_joint cold_plant 42; ladder M3.10_rgcn_sym_pair_joint cold_poll 42
if ! ls runs/M3.10_rgcn_sym_pair_joint/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] M3.10_rgcn_sym_pair_joint localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name M3.10_rgcn_sym_pair_joint --config "${ARMS[M3.10_rgcn_sym_pair_joint]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
fi
echo "[done] refine4 $(date +%H:%M)"
