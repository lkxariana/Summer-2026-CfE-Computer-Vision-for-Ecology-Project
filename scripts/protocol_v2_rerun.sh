#!/usr/bin/env bash
# Protocol v2 (negative pool = training pollinators): delete cold_poll / cold_both / warm bundles of every model that samples
# pollinator negatives and was written before the fix commit, then re-invoke the queues (their skip checks re-run only the
# missing bundles). Waits for the queues that touch those splits. GPU = $1.
set -u
GPU=$1
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
for L in battery warm_fix refine4 r3_system r31; do
  until grep -q "\[done\] $L" logs/$L.log 2>/dev/null; do sleep 180; done
done
FIX=$(git log --format=%H -1 --grep="negative-sampling pool")
echo "[protocol v2] fix commit $FIX $(date +%H:%M)"
DET="baseline_popularity baseline_cooccurrence baseline_abundance baseline_phenology_abundance baseline_congeneric baseline_svd_taxonomic baseline_nectar_ungated baseline_nectar_like baseline_tabicl"
for f in runs/*/*/cold_poll/val/s*/config.json runs/*/*/cold_both/val/s*/config.json runs/*/*/warm/val/s*/config.json; do
  [ -f "$f" ] || continue
  M=$(echo "$f" | cut -d/ -f2)
  case " $DET " in *" $M "*) continue;; esac
  C=$(/home/cher/miniconda3/envs/donuts/bin/python3 -c "import json;print(json.load(open('$f'))['git'])")
  if ! git merge-base --is-ancestor "$FIX" "$C" 2>/dev/null; then echo "[stale] $(dirname "$f")"; rm -rf "$(dirname "$f")"; fi
done
scripts/queue_battery.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|Traceback|Error|\[done\]"
scripts/queue_refine2.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|Traceback|Error|\[done\]"
scripts/queue_refine4.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|networks|Traceback|Error|\[done\]"
scripts/queue_r3_system.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|networks|Traceback|Error|\[done\]"
scripts/queue_r31.sh $GPU 2>&1 | grep -E "\[run\]|AUPR|networks|Traceback|Error|\[done\]"
echo "[done] protocol_v2 $(date +%H:%M)"
