#!/usr/bin/env bash
# Comparative (published-architecture) models through the ladder harness. Usage: queue_comparative.sh <gpu|cpu> <seed> [...]
set -u
DEV=$1; shift
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
run () { # name model config
  local NAME=$1 MODEL=$2 CFG=$3 SEED=$4
  if ls runs/$NAME/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $NAME s$SEED"; return; fi
  echo "[run] $NAME s$SEED $DEV $(date +%H:%M)"
  if [ "$DEV" = "cpu" ]; then
    CUDA_VISIBLE_DEVICES="" $P eval/run_ladder.py --model $MODEL --name $NAME --config "$CFG" --split cold_plant --seed $SEED --device cpu 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
  else
    CUDA_VISIBLE_DEVICES=$DEV $P eval/run_ladder.py --model $MODEL --name $NAME --config "$CFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
  fi
}
for SEED in "$@"; do
  if [ "$DEV" = "cpu" ]; then
    run baseline_antheia_spatial antheia_spatial "{\"seed\": $SEED}" $SEED
    run baseline_antheia_scalar  antheia_scalar  "{\"seed\": $SEED}" $SEED
    run baseline_pair_gbm        pair_gbm        "{\"seed\": $SEED}" $SEED
  else
    run baseline_widedeep  embednet '{"use_wide_affinity": true, "use_degree_offset": true}' $SEED
    run baseline_dcnv2     embednet '{"cross_layers": 2}' $SEED
    run baseline_pairnet   pairnet  "{\"seed\": $SEED}" $SEED
    run baseline_two_tower two_tower "{\"seed\": $SEED}" $SEED
  fi
done
echo "[done] comparative $DEV seeds $*"
