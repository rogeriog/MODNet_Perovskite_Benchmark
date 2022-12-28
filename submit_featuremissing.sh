#!/bin/bash
#SBATCH --job-name=
#SBATCH --time=05:00:00
#SBATCH --output=log_featurize.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=keira
#SBATCH --mem-per-cpu=2000
source ~/.bashrc

conda activate env_modnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
nproc=24 # $(nproc --all)
python3 featurize_missing_Perovskites.py >> log_featurize.txt
echo "done"
date
