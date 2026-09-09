#!/usr/bin/env bash
# Follow-ups on the R3 retriever: R6b concat fusion of presence, R6c learned co-presence bilinear head, R7 co-occurrence negatives.
# Each: cold_plant s42, local networks, cold_poll s42. Waits for r6. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] r6" logs/r6.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\""
declare -A ARMS
ARMS[M3.13_rgcn_sym_presconcat]="{$BASE, \"presence_input\": \"surface\", \"presence_fusion\": \"concat\"}"
ARMS[M3.14_rgcn_sym_bilinear]="{$BASE, \"pres_bilinear_rank\": 32}"
ARMS[M3.15_rgcn_sym_coocneg]="{$BASE, \"cooc_neg_frac\": 0.5}"
ladder () { # name split seed
  ls runs/$1/*/$2/val/s$3/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 $2 s$3"; return; }
  echo "[run] $1 $2 s$3 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split $2 --seed $3 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
for A in M3.13_rgcn_sym_presconcat M3.14_rgcn_sym_bilinear M3.15_rgcn_sym_coocneg; do
  ladder $A cold_plant 42
  if ! ls runs/$A/*/localnet/s42/metrics.json >/dev/null 2>&1; then
    echo "[run] $A localnet s42 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $A --config "${ARMS[$A]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
  fi
  ladder $A cold_poll 42
done
echo "[done] r67 $(date +%H:%M)"
