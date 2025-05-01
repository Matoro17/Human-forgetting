#!/bin/bash
#
#SBATCH -p batch # partition (queue)
#SBATCH --gpus 2 # change G by the number of GPU to use
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python3 -u models/dinov2/evaluate.py \
  --checkpoints-dir ./my_experiments/20250430_185012 \
  --data-root /home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl \
  --model-type vit_b \
  --device cuda