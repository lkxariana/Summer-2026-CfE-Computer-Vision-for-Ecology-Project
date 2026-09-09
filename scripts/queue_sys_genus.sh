#!/usr/bin/env bash
# Genus-profile tokens in the re-ranker, on the R3 retriever (the trees' lookup as a set; pair-level masking). Local networks first, then cold_plant s42. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\"}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true, \"genus_tokens\": 32, \"genus_loo\": \"pair\"}"
NAME=M4.1_system_sym_genus
if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
fi
if ! ls runs/$NAME/*/cold_plant/val/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME cold_plant s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split cold_plant --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
fi
echo "[done] sys_genus $(date +%H:%M)"
