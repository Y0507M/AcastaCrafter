#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --time=08:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=40GB            ## Allocated Memory
#SBATCH --cpus-per-task=16    ## Number of CPU cores
#SBATCH --job-name=download_obtain_diamond_pickaxe

# Downloading real gameplay files for obtain diamond pickaxe.
# https://github.com/openai/Video-Pre-Training?tab=readme-ov-file#contractor-demonstrations
python3 -u download.py
