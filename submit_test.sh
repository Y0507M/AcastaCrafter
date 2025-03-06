#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=20GB            ## Allocated Memory
#SBATCH --cpus-per-task 8     ## Number of CPU cores

module load ffmpeg # necessary for saving gifs to tensorboard on HPC3

export MINERL_JAVA_OPTS="-Xmx8G"
xvfb-run -a python test.py