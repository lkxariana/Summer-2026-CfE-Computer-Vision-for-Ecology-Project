#!/usr/bin/env bash
# A4: no-taxon retriever + explicit pair co-presence term (opportunity x affinity). cold_plant s42, cold_poll s42, and local networks
# scored two ways: full logit (A4) and affinity-only with the opportunity term zeroed at inference (A4o). Waits for sys_nocell. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] sys_a2" logs/sys_nocell.log 2>/dev/null; do sleep 120; done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\""
CFG="{$BASE}"; CFGO="{$BASE, \"pair_stat_at_inference\": false}"
NAME=A4_rgcn_notaxon_pair
for SPLIT in cold_plant cold_poll; do
  ls runs/$NAME/*/$SPLIT/val/s42/metrics.json >/dev/null 2>&1 && continue
  echo "[run] $NAME $SPLIT s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split $SPLIT --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
done
echo "[run] $NAME localnet s42 (full logit) $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $NAME --config "$CFG" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
echo "[run] ${NAME}_affinityonly localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name ${NAME}_affinityonly --config "$CFGO" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
echo "[done] a4 $(date +%H:%M)"
