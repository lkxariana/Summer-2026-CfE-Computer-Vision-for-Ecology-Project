#!/usr/bin/env bash
# A4 (no taxon + pair co-presence term) seeds 0/1: cold_plant, local networks, cold_poll -- it improved both tables on seed 42. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
CFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\"}"
NAME=A4_rgcn_notaxon_pair
for SEED in 0 1; do
  for SPLIT in cold_plant cold_poll; do
    ls runs/$NAME/*/$SPLIT/val/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] $SPLIT s$SEED"; continue; }
    echo "[run] $NAME $SPLIT s$SEED $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split $SPLIT --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
  done
  ls runs/${NAME}_affinityonly/*/localnet/s$SEED/metrics.json >/dev/null 2>&1 && { echo "[skip] localnet s$SEED"; continue; }
  echo "[run] ${NAME}_affinityonly localnet s$SEED $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name ${NAME}_affinityonly --config "${CFG%\}}, \"pair_stat_at_inference\": false}" --seed $SEED 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
done
echo "[done] a2_seeds $(date +%H:%M)"
