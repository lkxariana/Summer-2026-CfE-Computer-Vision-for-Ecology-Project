#!/usr/bin/env bash
# System v2 local-network seeds 0/1 (three-seed protocol for the final model). GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
for SEED in 0 1; do
  ls runs/M5.0_system_notaxon/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] localnet s$SEED"; continue; }
  echo "[run] M5.0_system_notaxon localnet s$SEED $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name M5.0_system_notaxon --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
done
echo "[done] sys_local $(date +%H:%M)"
