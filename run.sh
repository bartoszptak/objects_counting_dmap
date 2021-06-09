#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --job-name=visdrone-cc
#SBATCH --gres=gpu:1
#SBATCH --partition=tesla

source ~/.bashrc
source /home/users/putdron/miniconda3/etc/profile.d/conda.sh
conda activate cc 
cd /home/users/putdron/visdrone-cc/objects_counting_dmap
#python train.py -d visdrone -n UNet -e 150 -lr 0.003 
#python train.py -d visdrone -n UNet++_resnet34 -e 150 --mosaic --name visdrone_UNet++_resnet34_mosaic_sliced.pth
#python train.py -d visdrone -n UNet++_resnet34 --epochs 150 --mosaic --flow
python train.py -d visdrone -n UNet++_resnet34 --epochs 100 --flow dis2 --mosaic
