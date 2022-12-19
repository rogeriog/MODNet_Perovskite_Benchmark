#!/bin/bash
#SBATCH --job-name=MEGNet16
#SBATCH --time=1-00:00:00
#SBATCH --output=log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=keira
#SBATCH --mem-per-cpu=2000
source ~/.bashrc

conda activate env_modnetmod
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
nproc=12 # $(nproc --all)
python3 get_megnetfeats16.py >> log.txt
echo "done"
date
