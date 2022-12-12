#!/bin/bash
#SBATCH --job-name=SOAP_perovskites
#SBATCH --time=8:00:00
#SBATCH --output=log.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
source ~/.bashrc

conda activate env_modnetmod
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
#python3 SOAP_featurization.py >> log.txt
python3 -u SOAP_concatenate.py >> log.txt
echo "done"
date
