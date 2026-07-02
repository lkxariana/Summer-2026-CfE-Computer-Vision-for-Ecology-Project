#!/bin/bash
#SBATCH --account=bdbl-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=180g
#SBATCH --time=03:00:00
#SBATCH --job-name=covariate
#SBATCH --output=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity/outputs/logs/covariate_%j.log
#SBATCH --error=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity/outputs/logs/covariate_%j.err
# gpu_a100 full? resubmit with: sbatch --partition=gpu --account=bgxd-tgirails <this>
set -euo pipefail
export PPE_OUT_DIR=/projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/Part2_PhenoField_Outputs
cd /projects/bdbl/cherd/Summer-2026-CfE-Computer-Vision-for-Ecology-Project/Plant_Pollinator_Initial_Analysis/PhenoField_Opportunity
mkdir -p outputs/logs
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pheno
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python run_covariate_eval.py --feat "$PPE_OUT_DIR/data/grid_ppel_features_all.npz" --tag ppel_ci --seeds 0,1,2,3,4
