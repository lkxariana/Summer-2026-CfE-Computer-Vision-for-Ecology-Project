#!/usr/bin/env bash
# System v2 (R3 + re-ranker) on the pollinator-side splits: s42 where missing, then seeds 0/1, so the final table has 3 seeds everywhere. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
NAME=M5.0_system_notaxon
sys () { # split seed
  ls runs/$NAME/*/$1/val/s$2/metrics.json >/dev/null 2>&1 && { echo "[skip] $NAME $1 s$2"; return; }
  echo "[run] $NAME $1 s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split $1 --seed $2 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
}
for SPLIT in cold_both cold_poll warm; do sys $SPLIT 42; done
for SEED in 0 1; do for SPLIT in cold_poll cold_both warm; do sys $SPLIT $SEED; done; done
# R3 retriever seeds 0/1 on cold_both and warm for the ablation row
CFG="$RCFG"
for SEED in 0 1; do for SPLIT in cold_both warm; do
  ls runs/A2_rgcn_sym_notaxon/*/$SPLIT/val/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] A2 $SPLIT s$SEED"; continue; }
  echo "[run] A2_rgcn_sym_notaxon $SPLIT s$SEED $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name A2_rgcn_sym_notaxon --config "$CFG" --split $SPLIT --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
done; done
echo "[done] sys_seeds $(date +%H:%M)"
