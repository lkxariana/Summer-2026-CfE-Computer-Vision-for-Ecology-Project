#!/bin/bash
#SBATCH --account=bdbl-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=180g
#SBATCH --time=03:00:00
#SBATCH --job-name=extract_ppel
#SBATCH --output=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity/outputs/logs/extract_ppel_%j.log
#SBATCH --error=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity/outputs/logs/extract_ppel_%j.err
# gpu_a100 full? resubmit with: sbatch --partition=gpu --account=bgxd-tgirails <this>
set -euo pipefail
export PPE_OUT_DIR=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs
cd /projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity
mkdir -p outputs/logs
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pheno
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python build_prism_weekly.py
python extract_ppel_field.py --cells all
