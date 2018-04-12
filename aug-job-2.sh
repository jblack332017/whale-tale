#!/bin/bash

#SBATCH --time=03:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=32768M   # memory per CPU core
#SBATCH -J "large-aug-2"   # job name
#SBATCH --mail-user=josh.m.black.work@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/2/7
module load cuda/8.0
module load cudnn/5.1_cuda-8.0

./testmodel.sh
