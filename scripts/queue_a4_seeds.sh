#!/usr/bin/env bash
# A4 seeds 0/1: cold_plant, cold_poll, within-site full and affinity-only. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\""
CFG="{$BASE}"; CFGO="{$BASE, \"pair_stat_at_inference\": false}"
NAME=A4_rgcn_notaxon_pair
for SEED in 0 1; do
  for SPLIT in cold_plant cold_poll; do
    ls runs/$NAME/*/$SPLIT/val/s$SEED/metrics.json >/dev/null 2>&1 && continue
    echo "[run] $NAME $SPLIT s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split $SPLIT --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
  done
  ls runs/$NAME/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 || { echo "[run] $NAME localnet s$SEED $(date +%H:%M)"; CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $NAME --config "$CFG" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"; }
  ls runs/${NAME}_affinityonly/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 || { echo "[run] ${NAME}_affinityonly localnet s$SEED $(date +%H:%M)"; CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name ${NAME}_affinityonly --config "$CFGO" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"; }
done
echo "[done] a4_seeds $(date +%H:%M)"
