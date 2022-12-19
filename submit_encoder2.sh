#!/bin/bash
#SBATCH --job-name=AutoEncoder2_MODNetPerovsk
#SBATCH --time=1-00:00:00
#SBATCH --output=log2.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
## #SBATCH --nodelist=mb-rom[101,102,103,104]
source ~/.bashrc
module load CUDA cuDNN/8.0.4.30-CUDA-11.1.1 
# TensorFlow/2.5.0-fosscuda-2020b
#export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_megnetgpu/lib/"
CUDA_DIR=/home/ucl/modl/rgouvea/anaconda3/envs/env_modnetmod/lib/
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_modnetmod/lib/" 
# /home/ucl/modl/rgouvea/xla/nvvm/libdevice"

conda activate env_modnetmod
echo "start"
date
python3 -u autoencoderMODNetFeats2.py >> log2.txt
date
echo "done"

