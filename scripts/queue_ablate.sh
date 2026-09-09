#!/usr/bin/env bash
# Structural ablations of the R3 retriever (plan §4 controls): no cell x month nodes, no taxon nodes, month-collapsed cells.
# Each: cold_plant s42, local networks, cold_poll s42. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
BASE="\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false, \"anchor_side\": \"both\""
declare -A ARMS
ARMS[A1_rgcn_sym_nocell]="{$BASE, \"use_cell_nodes\": false}"
ARMS[A2_rgcn_sym_notaxon]="{$BASE, \"use_taxon_nodes\": false}"
ARMS[A3_rgcn_sym_monthcollapsed]="{$BASE, \"month_collapsed\": true}"
ladder () { # name split seed
  ls runs/$1/*/$2/val/s$3/metrics.json >/dev/null 2>&1 && { echo "[skip] $1 $2 s$3"; return; }
  echo "[run] $1 $2 s$3 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name $1 --config "${ARMS[$1]}" --split $2 --seed $3 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "AUPR|Traceback|Error"
}
for A in A3_rgcn_sym_monthcollapsed A1_rgcn_sym_nocell A2_rgcn_sym_notaxon; do
  ladder $A cold_plant 42
  if ! ls runs/$A/*/localnet/s42/metrics.json >/dev/null 2>&1; then
    echo "[run] $A localnet s42 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model rgcn --name $A --config "${ARMS[$A]}" --seed 42 2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "networks|Traceback|Error"
  fi
  ladder $A cold_poll 42
done
echo "[done] ablate $(date +%H:%M)"
