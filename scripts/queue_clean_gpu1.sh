#!/usr/bin/env bash
# Clean-protocol re-runs of OUR models on the pollinator-side splits, on GPU 1 as soon as the attention control finishes
# (protocol_v2_rerun.sh handles the baselines on GPU 0; both use skip checks, so whichever finishes an arm first wins). GPU = $1.
set -u
GPU=$1
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
until grep -q "\[done\] attn" logs/attn.log 2>/dev/null; do sleep 120; done
FIX=$(git log --format=%H -1 --grep="negative-sampling pool")
for M in M3.7_rgcn_sym M4.0_system_sym M3.11_rgcn_sym_warmres M3.1_rgcn M2.12_fusion_on_rgcn; do
  for f in runs/$M/*/cold_poll/val/s*/config.json runs/$M/*/cold_both/val/s*/config.json runs/$M/*/warm/val/s*/config.json; do
    [ -f "$f" ] || continue
    C=$(/home/cher/miniconda3/envs/donuts/bin/python3 -c "import json;print(json.load(open('$f'))['git'])")
    git merge-base --is-ancestor "$FIX" "$C" 2>/dev/null || { echo "[stale] $(dirname "$f")"; rm -rf "$(dirname "$f")"; }
  done
done
scripts/queue_r3_system.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|networks|Traceback|Error|\[done\]"
scripts/queue_refine2.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|Traceback|Error|\[done\]"
scripts/queue_r31.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|Traceback|Error|\[done\]"
echo "[done] clean_gpu1 $(date +%H:%M)"
