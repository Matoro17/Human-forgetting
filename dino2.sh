#!/bin/bash
#
#SBATCH -p batch # partition (queue)
#SBATCH --gpus 2 # change G by the number of GPU to use
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python3 -u /home/gabrielazevedo/Human-forgetting/models/dinov2/train.py
