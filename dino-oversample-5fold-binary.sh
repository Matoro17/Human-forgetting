#!/bin/bash
#
#SBATCH -p batch # partition (queue)
#SBATCH --gpus 1 # change G by the number of GPU to use
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python3 -u /home/gabrielazevedo/Human-forgetting/models/SSL_ova_fixed_kfold.py --data_dir "/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/baseline" --results_dir "./results_dino_binary/oversample"
