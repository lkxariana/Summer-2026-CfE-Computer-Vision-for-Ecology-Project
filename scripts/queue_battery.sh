#!/usr/bin/env bash
# Split battery (plan S1-S4): cold_poll, cold_both, warm for the frozen system, its retriever, and the comparative set.
# Usage: queue_battery.sh <gpu>. Seed 42 for everything first, then seeds 0/1 for the system and R-GCN.
set -u
GPU=$1; P=/home/cher/miniconda3/envs/donuts/bin/python3
cd /scratch/cher/Summer-2026-CfE-Computer-Vision-for-Ecology-Project
F=/scratch/cher/antheia-data/pollinator_sdm/joint_field/field_v2
RCFG="{\"field_dir\": \"$F\", \"softmax_weight\": 1.0, \"bce_weight\": 0.5, \"head_type\": \"elementwise\", \"use_degree_heads\": false}"
FCFG="{\"field_dir\": \"$F\", \"identity_only\": true}"
done_ () { ls runs/$1/*/$2/val/s$3/metrics.json >/dev/null 2>&1; }
ladder () { # name model config split seed
  done_ $1 $4 $5 && { echo "[skip] $1 $4 s$5"; return; }
  echo "[run] $1 $4 s$5 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_ladder.py --model $2 --name $1 --config "$3" --split $4 --seed $5 2>&1 | grep -v Warning | grep -E "AUPR|Traceback|Error"
}
system () { # split seed
  done_ M2.12_fusion_on_rgcn $1 $2 && { echo "[skip] system $1 s$2"; return; }
  echo "[run] system $1 s$2 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU $P eval/run_fusion.py --name M2.12_fusion_on_rgcn --retriever-model rgcn --retriever-config "$RCFG" --config "$FCFG" --split $1 --seed $2 2>&1 | grep -v Warning | grep -E "recall@|AUPR|Traceback|Error"
}
for SPLIT in cold_poll cold_both warm; do
  ladder M3.1_rgcn rgcn "$RCFG" $SPLIT 42
  system $SPLIT 42
  for M in popularity cooccurrence abundance phenology_abundance congeneric svd_taxonomic nectar_ungated nectar_like ours_gbm routed; do
    ladder baseline_$M $M '{}' $SPLIT 42
  done
  ladder baseline_antheia_spatial antheia_spatial '{"seed": 42}' $SPLIT 42
  ladder baseline_antheia_scalar  antheia_scalar  '{"seed": 42}' $SPLIT 42
  ladder baseline_pair_gbm        pair_gbm        '{"seed": 42}' $SPLIT 42
  ladder baseline_widedeep  embednet '{"use_wide_affinity": true, "use_degree_offset": true}' $SPLIT 42
  ladder baseline_dcnv2     embednet '{"cross_layers": 2}' $SPLIT 42
  ladder baseline_pairnet   pairnet  '{"seed": 42}' $SPLIT 42
  ladder baseline_two_tower two_tower '{"seed": 42}' $SPLIT 42
  ladder M1.0_reference embednet '{"epochs": 25, "use_genus_context": false, "use_tier_head": false}' $SPLIT 42
done
for SEED in 0 1; do for SPLIT in cold_poll cold_both warm; do
  ladder M3.1_rgcn rgcn "$RCFG" $SPLIT $SEED
  system $SPLIT $SEED
done; done
echo "[done] battery $(date +%H:%M)"
