#!/bin/bash
#SBATCH --job-name=MEGNetFeats_perovskites
#SBATCH --time=4:00:00
#SBATCH --output=log.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
source ~/.bashrc

## module load CUDA cuDNN/8.0.4.30-CUDA-11.1.1
# TensorFlow/2.5.0-fosscuda-2020b
#export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_megnetgpu/lib/"
## CUDA_DIR=/home/ucl/modl/rgouvea/anaconda3/envs/env_modnetmod/lib/
## export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_modnetmod/lib/"

conda activate env_modnetmod
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
#python3 SOAP_featurization.py >> log.txt
python3 -u MEGNetFeaturization.py >> log.txt
echo "done"
date
