module load python/2/7
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
python getLabels.py input > input.log
python getLabels.py input augmented > inaug.log
python getLabels.py input augmented-no-zoom > inaugnz.log
