#!/bin/bash

#SBATCH --time=01:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=65536M   # memory per CPU core
#SBATCH -J "withunlabeled"   # job name
#SBATCH --mail-user=spencer.smith2012@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/2/7
module load cuda/8.0
module load cudnn/5.1_cuda-8.0

python ../whale-tale.py withlabel.csv 6 ../test ../input ../augmented-correct-rotation > withlabel.log
