#!/usr/bin/env bash
# System on the A2 retriever (symmetric rehearsal, no taxon nodes) + identity re-ranker: cold_plant s42, local networks, cold_poll, cold_both, warm s42. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"use_cell_nodes\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
NAME=M5.1_system_nocell_notaxon
sys () { # split seed
  ls runs/$NAME/*/$1/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $NAME $1 s$2"; return; }
  echo "[run] $NAME $1 s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split $1 --seed $2 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
}
sys cold_plant 42
if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] $NAME localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
fi
for SPLIT in cold_poll cold_both warm; do sys $SPLIT 42; done
for SEED in 0 1; do sys cold_plant $SEED; done
echo "[done] sys_a2 $(date +%H:%M)"
