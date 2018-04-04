module load python/2/7
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
python model.py input > input.log
python model.py input augmented > inaug.log
python model.py input augmented-no-zoom > inaugnz.log
