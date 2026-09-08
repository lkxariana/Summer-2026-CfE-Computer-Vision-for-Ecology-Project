#!/usr/bin/env bash
# Local-network completion for the comparative set. Usage: queue_localnet.sh cpu | queue_localnet.sh <gpu> <seed>...
set -u
DEV=$1; shift
P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
run () { local NAME=$1 MODEL=$2 CFG=$3 SEED=$4
  if ls runs/$NAME/*/localnet/s$SEED/metrics.json >/dev/null 2>&1; then echo "[skip] $NAME s$SEED"; return; fi
  echo "[run] $NAME s$SEED $DEV $(date +%H:%M)"
  if [ "$DEV" = "cpu" ]; then CUDA_VISIBLE_DEVICES="" $P eval/run_localnets.py --model $MODEL --name $NAME --config "$CFG" --seed $SEED --device cpu 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"
  else CUDA_VISIBLE_DEVICES=$DEV $P eval/run_localnets.py --model $MODEL --name $NAME --config "$CFG" --seed $SEED 2>&1 | grep -v Warning | grep -E "networks|Traceback|Error"; fi
}
if [ "$DEV" = "cpu" ]; then
  for M in popularity cooccurrence abundance phenology_abundance congeneric svd_taxonomic antheia_spatial antheia_scalar pair_gbm ours_gbm routed; do run baseline_$M $M '{}' 42; done
else
  for SEED in "$@"; do
    run M1.0_reference embednet '{}' $SEED
    run baseline_widedeep embednet '{"use_wide_affinity": true, "use_degree_offset": true}' $SEED
    run baseline_dcnv2 embednet '{"cross_layers": 2}' $SEED
  done
fi
echo "[done] localnet $DEV $*"
