#!/usr/bin/env bash
# Rebuild the network products from the raw records: edges -> canonical nodes -> modelled universe -> splits -> local networks.
# Feature caches (text/image embeddings, surfaces, tokens, fields) are built by the scripts in pipelines/features/ and
# pipelines/sdm/, pipelines/ppe/ (GPU, hours); see pipelines/README.md. Inputs and outputs live under $ANTHEIA_DATA.
set -o pipefail
PY=${PY:-/home/cher/miniconda3/envs/donuts/bin/python3}
mkdir -p logs
for step in build_edges canonicalize_nodes build_modelled_universe build_splits build_splits_battery extract_local_networks; do
  echo "=== $step $(date -Is) ===" | tee -a logs/build.log
  $PY pipelines/network/$step.py 2>&1 | grep -viE "^ *$|warning" | tee -a logs/build.log || { echo "$step failed"; exit 1; }
done
echo "=== tests $(date -Is) ===" | tee -a logs/build.log
$PY tests/test_edges.py 2>&1 | tee -a logs/build.log
$PY tests/test_splits.py 2>&1 | tee -a logs/build.log
