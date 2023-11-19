#!/bin/bash

#SBATCH -N 1 
#SBATCH --ntasks-per-node=16 #number of cores per node
#SBATCH --time=4-00:00:00 
#SBATCH --job-name=LipNet-uni  #change name of ur job
#SBATCH --output=output.uni  #change name of ur output file
#SBATCH --partition=gpu  #there are various partition. U can change various GPUs
#SBATCH --gres=gpu:2 #same as above
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t22104@students.iitmandi.ac.in

# Load module
module load DL-Conda_3.7

# activate environment 
source /home/apps/DL/DL-CondaPy3.7/bin/activate torch
cd $SLURM_SUBMIT_DIR

CUDA_VISIBLE_DEVICES=0,1 python train-uni.py
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 train.py
