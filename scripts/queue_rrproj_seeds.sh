#!/usr/bin/env bash
# M6.1 (re-ranker tokens = retriever's projected names) seeds 0/1. Usage: queue_rrproj_seeds.sh <gpu> <cold_plant|local>
set -u
GPU=$1; MODE=$2; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RB="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\""
RCFG="{$RB}"; RCFGO="{$RB, \"pair_stat_at_inference\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true, \"id_source\": \"retriever_proj\"}"
NAME=M6.1_system_rrproj
for SEED in 0 1; do
  if [ "$MODE" = "cold_plant" ]; then
    ls runs/$NAME/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1 && continue
    echo "[run] $NAME cold_plant s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split cold_plant --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
  else
    ls runs/$NAME/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 && continue
    echo "[run] $NAME localnet s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFGO" --config "$FCFG" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
  fi
done
echo "[done] rrproj_$MODE $(date +%H:%M)"
