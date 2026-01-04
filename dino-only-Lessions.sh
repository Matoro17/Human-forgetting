#!/bin/bash
#
#SBATCH -p batch # partition (queue)
#SBATCH --gpus 2 # change G by the number of GPU to use
#SBATCH -o 2026-slurm.%N.%j.out # STDOUT
#SBATCH -e 2026-slurm.%N.%j.err # STDERR

python3 -u /home/gabrielazevedo/Human-forgetting/models/experiment_multiclass_onlylessions.py
