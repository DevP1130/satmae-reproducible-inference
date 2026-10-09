#!/bin/bash
#SBATCH -J satmae-build-env
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load anaconda3
export CONDA_PKGS_DIRS=$HOME/scratch/conda_pkgs
conda env create -f env/environment.yml -p $HOME/scratch/envs/satmae
source activate $HOME/scratch/envs/satmae
python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'timm', timm.__version__)"
conda env export -p $HOME/scratch/envs/satmae > env/environment.lock.yml
pip freeze > env/pip-freeze.txt

