#!/usr/bin/env bash
# Refinement arms on the R-GCN retriever (frozen objective): R1 warm residual, R2 direct genus-partner relations, R1+R2.
# Pass 1: seed 42 + local networks for every arm (early read); pass 2: seeds 0/1. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false"
declare -A ARMS
ARMS[M3.4_rgcn_genus_edges]="{$BASE, \"genus_edges\": true}"
ARMS[M3.3_rgcn_warmres]="{$BASE, \"warm_residual\": true}"
ARMS[M3.5_rgcn_genus_warmres]="{$BASE, \"genus_edges\": true, \"warm_residual\": true}"
ORDER="M3.4_rgcn_genus_edges M3.3_rgcn_warmres M3.5_rgcn_genus_warmres"
ladder () { # name seed
  ls runs/$1/*/cold_plant/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 s$2"; return; }
  echo "[run] $1 cold_plant s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split cold_plant --seed $2 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
local_ () { # name
  ls runs/$1/*/localnet/*/s42/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 localnet"; return; }
  echo "[run] $1 localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $1 --config "${ARMS[$1]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|AUPR|Traceback|Error"
}
for A in $ORDER; do ladder $A 42; local_ $A; done
for SEED in 0 1; do for A in $ORDER; do ladder $A $SEED; done; done
echo "[done] refine $(date +%H:%M)"
