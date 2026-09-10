#!/usr/bin/env bash
# M6.0 factorised system: seeds 0/1 on cold_poll / cold_both / warm (GPU 0) or local networks (GPU 1). Usage: queue_m6_seeds.sh <gpu> <splits|local>
set -u
GPU=$1; MODE=$2; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\"}"
RCFGO="${RCFG%\}}, \"pair_stat_at_inference\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
NAME=M6.0_system_factorised
if [ "$MODE" = "splits" ]; then
  for SEED in 0 1; do for SPLIT in cold_poll cold_both warm; do
    ls runs/$NAME/*/$SPLIT/val/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] $SPLIT s$SEED"; continue; }
    echo "[run] $NAME $SPLIT s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split $SPLIT --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
  done; done
else
  for SEED in 0 1; do
    ls runs/$NAME/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] localnet s$SEED"; continue; }
    echo "[run] $NAME localnet s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFGO" --config "$FCFG" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
  done
fi
echo "[done] m6_seeds_$MODE $(date +%H:%M)"
