#!/usr/bin/env bash
# System on the R3 retriever (symmetric anchors): identity re-ranker on top, cold_plant 3 seeds + local networks (fusion) + battery s42.
# Starts only when logs/r3_system.go exists (created after R3 seeds 0/1 confirm). GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until [ -f logs/r3_system.go ]; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\"}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
NAME=M4.0_system_sym
sys () { # split seed
  ls runs/$NAME/*/$1/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $NAME $1 s$2"; return; }
  echo "[run] $NAME $1 s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split $1 --seed $2 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error"
}
for SEED in 42 0 1; do sys cold_plant $SEED; done
if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
fi
for SPLIT in cold_poll cold_both warm; do sys $SPLIT 42; done
echo "[done] r3_system $(date +%H:%M)"
