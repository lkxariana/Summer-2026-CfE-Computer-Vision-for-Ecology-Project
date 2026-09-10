#!/usr/bin/env bash
# Design ablations on the final model. Usage: queue_design.sh <gpu> <retriever|reranker>
#  retriever: K1 = kingdom-specific text projections in the A4 retriever (cold_plant s42, within-site affinity-only, cold_poll s42)
#  reranker : M6.1 = re-ranker tokens from the retriever's projected names; M6.2 = from the retriever's graph outputs h (cold_plant s42, within-site)
set -u
GPU=$1; MODE=$2; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RB="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\", \"use_taxon_nodes\": false, \"pair_stat\": \"joint\""
if [ "$MODE" = "retriever" ]; then
  NAME=K1_rgcn_kingdomproj; CFG="{$RB, \"text_proj\": \"kingdom\"}"; CFGO="{$RB, \"text_proj\": \"kingdom\", \"pair_stat_at_inference\": false}"
  for SPLIT in cold_plant cold_poll; do
    ls runs/$NAME/*/$SPLIT/val/s42/metrics.json >/dev/null 2>&1 && continue
    echo "[run] $NAME $SPLIT s42 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $NAME --config "$CFG" --split $SPLIT --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
  done
  echo "[run] ${NAME}_affinityonly localnet s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name ${NAME}_affinityonly --config "$CFGO" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
else
  RCFG="{$RB}"; RCFGO="{$RB, \"pair_stat_at_inference\": false}"
  for SRC in retriever_proj retriever_h; do
    NAME=M6.${SRC}_system; [ "$SRC" = retriever_proj ] && NAME=M6.1_system_rrproj || NAME=M6.2_system_rrh
    FCFG="{\"field_dir\": \"$F\", \"identity_only\": true, \"id_source\": \"$SRC\"}"
    if ! ls runs/$NAME/*/cold_plant/val/s42/metrics.json >/dev/null 2>&1; then
      echo "[run] $NAME cold_plant s42 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split cold_plant --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "recall@|AUPR|Traceback|Error"
    fi
    if ! ls runs/$NAME/*/localnet/s42/metrics.json >/dev/null 2>&1; then
      echo "[run] $NAME localnet s42 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name $NAME --retriever-model rgcn --retriever-config "$RCFGO" --config "$FCFG" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
    fi
  done
fi
echo "[done] design_$MODE $(date +%H:%M)"
