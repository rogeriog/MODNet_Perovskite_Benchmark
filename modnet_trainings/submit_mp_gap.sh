#!/bin/bash
#SBATCH --job-name=matbench_mp_gap
#SBATCH --time=2-00:00:00
#SBATCH --output=log_mp_gap.txt
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
python3 run_benchmark.py --task matbench_mp_gap --n_jobs $nproc >> log_mp_gap.txt
echo "done"
date
