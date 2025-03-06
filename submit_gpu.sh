#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=10:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 8    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

module load ffmpeg # necessary for saving gifs to tensorboard on HPC3

export MINERL_JAVA_OPTS="-Xmx8G"
xvfb-run -a python main.py