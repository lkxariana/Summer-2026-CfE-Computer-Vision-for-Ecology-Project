#!/usr/bin/env bash
# S1 retriever arms (plan §2), two phases. Usage: queue_s1.sh <gpu> <phase A|B> <seed> [<seed> ...]
# Phase A: the loss sweep (reference + softmax/bce weights). Phase B: heads on the phase-A winner (BASE below).
set -u
GPU=$1; PHASE=$2; shift 2
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
declare -A ARMS=(
  ["M1.0_reference"]='{}'
  ["M1.1_pooled"]='{"softmax_weight": 0.0, "bce_weight": 1.0}'
  ["M1.2a_pooled_sm0.25"]='{"softmax_weight": 0.25, "bce_weight": 1.0}'
  ["M1.2b_sm1_bce1"]='{"softmax_weight": 1.0, "bce_weight": 1.0}'
  ["M1.2c_sm1_bce0.25"]='{"softmax_weight": 1.0, "bce_weight": 0.25}'
  ["M1.2d_sm1_bce2"]='{"softmax_weight": 1.0, "bce_weight": 2.0}'
)
BASE='"softmax_weight": 1.0, "bce_weight": 0.5'   # phase-A winner; edit before launching phase B
ARMS["M1.3_degheads"]="{$BASE, \"use_degree_heads\": true}"
ARMS["M1.4_degheads_pu"]="{$BASE, \"use_degree_heads\": true, \"pu_prior\": 0.0013}"
ARMS["M1.5_degheads_bilinear"]="{$BASE, \"use_degree_heads\": true, \"head_type\": \"concat_bilinear\"}"
ARMS["M1.5b_bilinear_only"]="{$BASE, \"head_type\": \"concat_bilinear\"}"
if [ "$PHASE" = "A" ]; then
  LIST="M1.0_reference M1.1_pooled M1.2a_pooled_sm0.25 M1.2b_sm1_bce1 M1.2c_sm1_bce0.25 M1.2d_sm1_bce2"
else
  LIST="M1.3_degheads M1.4_degheads_pu M1.5_degheads_bilinear M1.5b_bilinear_only"
fi
for SEED in "$@"; do
  for ARM in $LIST; do
    if ls runs/$ARM/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $ARM s$SEED"; continue; fi
    echo "[run] $ARM s$SEED gpu$GPU $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model embednet --name $ARM --config "${ARMS[$ARM]}" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
  done
done
echo "[done] phase $PHASE seeds $*"
