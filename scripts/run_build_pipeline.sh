#!/usr/bin/env bash
# Build -> EDA -> tests, chained. Each stage logs to data/.
set -o pipefail
PY=/home/cher/miniconda3/envs/donuts/bin/python3
mkdir -p data
echo "=== BUILD $(date -Is) ==="
$PY scripts/build_edges.py 2>&1 | grep -viE "^ *$|warning" | tee data/build.log
BUILD=$?
echo "=== EDA $(date -Is) ==="
$PY eval/eda_edges.py 2>&1 | grep -viE "warning" | tee data/eda.log
echo "=== TESTS $(date -Is) ==="
$PY tests/test_edges.py 2>&1 | tee data/tests.log
echo "=== DONE $(date -Is) build_exit=$BUILD ==="
