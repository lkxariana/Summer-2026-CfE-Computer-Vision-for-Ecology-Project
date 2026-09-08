#!/usr/bin/env bash
# R4 explicit pair co-presence statistic in the head (joint, with space/time marginal controls) and R5 degree encoding.
# Waits for refine2 on the same GPU. Pass 1: seed 42 + local networks per arm; pass 2: seeds 0/1 for joint and degree. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] refine2" logs/refine2.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false"
declare -A ARMS
ARMS[M3.8_rgcn_pair_joint]="{$BASE, \"pair_stat\": \"joint\"}"
ARMS[M3.9_rgcn_degree_enc]="{$BASE, \"degree_encoding\": true}"
ARMS[M3.8s_rgcn_pair_space]="{$BASE, \"pair_stat\": \"space\"}"
ARMS[M3.8t_rgcn_pair_time]="{$BASE, \"pair_stat\": \"time\"}"
ORDER="M3.8_rgcn_pair_joint M3.9_rgcn_degree_enc M3.8s_rgcn_pair_space M3.8t_rgcn_pair_time"
ladder () { # name seed
  ls runs/$1/*/cold_plant/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 s$2"; return; }
  echo "[run] $1 cold_plant s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split cold_plant --seed $2 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
local_ () { # name
  ls runs/$1/*/localnet/s42/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 localnet"; return; }
  echo "[run] $1 localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $1 --config "${ARMS[$1]}" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
}
for A in $ORDER; do ladder $A 42; done
for A in M3.8_rgcn_pair_joint M3.9_rgcn_degree_enc; do local_ $A; done
for SEED in 0 1; do for A in M3.8_rgcn_pair_joint M3.9_rgcn_degree_enc; do ladder $A $SEED; done; done
echo "[done] refine3 $(date +%H:%M)"
