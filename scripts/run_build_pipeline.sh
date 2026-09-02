#!/usr/bin/env bash
# Build -> EDA -> tests, chained. Each stage logs to artifacts/v2/.
set -o pipefail
PY=/home/cher/miniconda3/envs/donuts/bin/python3
mkdir -p artifacts/v2
echo "=== BUILD $(date -Is) ==="
$PY scripts/build_edges.py 2>&1 | grep -viE "^ *$|warning" | tee artifacts/v2/build.log
BUILD=$?
echo "=== EDA $(date -Is) ==="
$PY eval/eda_edges_v2.py 2>&1 | grep -viE "warning" | tee artifacts/v2/eda.log
echo "=== TESTS $(date -Is) ==="
$PY tests/test_edges_v2.py 2>&1 | tee artifacts/v2/tests.log
echo "=== DONE $(date -Is) build_exit=$BUILD ==="
