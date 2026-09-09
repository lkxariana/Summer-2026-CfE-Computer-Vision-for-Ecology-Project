#!/usr/bin/env bash
# After the battery: delete warm bundles written before filtered ranking (metrics.json without n_excluded) and re-run them. GPU = $1.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] battery" logs/battery.log 2>/dev/null; do sleep 120; done
for f in runs/*/*/warm/val/s*/metrics.json; do
  grep -q n_excluded "$f" || { echo "[stale] $f"; rm -rf "$(dirname "$f")"; }
done
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
if ! ls runs/M3.1_rgcn/*/warm/val/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] M3.1_rgcn warm s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model rgcn --name M3.1_rgcn --config "$RCFG" --split warm --seed 42 2>&1 | grep -v Warning | grep -E "AUPR|excluded|Traceback|Error"
fi
if ! ls runs/M2.12_fusion_on_rgcn/*/warm/val/s42/metrics.json >/dev/null 2>&1; then
  echo "[run] system warm s42 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name M2.12_fusion_on_rgcn --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split warm --seed 42 2>&1 | grep -v Warning | grep -E "recall@|AUPR|excluded|Traceback|Error"
fi
# any baseline that ran on warm before the fix: re-run through the battery script (its skip checks see the deleted bundles)
scripts/queue_battery.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|Traceback|Error|\[done\]"
echo "[done] warm_fix $(date +%H:%M)"
