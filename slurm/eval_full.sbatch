#!/bin/bash
#SBATCH --job-name=satmae-eval-full
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs results

echo "== job: eval_full  job_id: ${SLURM_JOB_ID}  node: $(hostname) =="
date

echo "== nvidia-smi ==" | tee "logs/eval_full_${SLURM_JOB_ID}_nvidia-smi.txt"
nvidia-smi | tee -a "logs/eval_full_${SLURM_JOB_ID}_nvidia-smi.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/satmae

echo "== torch/CUDA versions ==" | tee "logs/eval_full_${SLURM_JOB_ID}_versions.txt"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())" \
  | tee -a "logs/eval_full_${SLURM_JOB_ID}_versions.txt"

START_TIME=$(date +%s)

python scripts/evaluate.py \
  --csv data/val_prepared.csv \
  --data-root data \
  --checkpoint weights/finetune-vit-base-e7.pth \
  --batch-size 64 \
  --num-workers 8 \
  --out-dir results

END_TIME=$(date +%s)
echo "Wall time (s): $((END_TIME - START_TIME))" | tee "logs/eval_full_${SLURM_JOB_ID}_walltime.txt"

echo "== sacct =="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize \
  | tee "logs/eval_full_${SLURM_JOB_ID}_sacct.txt"

date
echo "== done =="
