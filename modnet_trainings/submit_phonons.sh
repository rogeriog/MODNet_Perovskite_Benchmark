#!/bin/bash
#SBATCH --job-name=matbench_phonons
#SBATCH --time=1-00:00:00
#SBATCH --output=log_phonons.txt
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
python3 run_benchmark.py --task matbench_phonons --n_jobs $nproc >> log_phonons.txt
echo "done"
date
