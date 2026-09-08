#!/usr/bin/env bash
# M2.8: fusion re-ranker on the genus-routed tree retriever (booster candidates + neural residual re-ranking), then the
# embedding-model local-network re-run (warm/cold columns), then M3.1 R-GCN. GPU given as $1.
set -u
GPU=$1
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
CFG='{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2", "identity_only": true}'
for SEED in 42 0 1; do
  if ls runs/M2.8_fusion_routed/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] M2.8 s$SEED"; continue; fi
  echo "[run] M2.8_fusion_routed s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name M2.8_fusion_routed --retriever-model routed --config "$CFG" --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error"
done
echo "[run] M2.8 localnet s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets_fusion.py --name M2.8_fusion_routed --retriever-model routed --config "$CFG" --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
rm -rf runs/M1.0_reference/*/localnet/s42
echo "[run] M1.0 localnet rerun s42 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU $P eval/run_localnets.py --model embednet --name M1.0_reference --seed 42 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
for SEED in 42 0 1; do
  if ls runs/M3.1_rgcn/*/cold_plant/val/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] M3.1 s$SEED"; continue; fi
  echo "[run] M3.1_rgcn s$SEED gpu$GPU $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name M3.1_rgcn --config '{"field_dir": "/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2"}' --split cold_plant --seed $SEED 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
done
echo "[done] M2.8 + M3.1"
