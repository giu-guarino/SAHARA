#!/usr/bin/bash
echo "Launching job $OAR_JOBID on oarprint gpunb gpus on host oarprint host"
module load conda/2021.11-python3.9
module load cuda/11.0
eval "$(conda shell.bash hook)"
conda activate sshda-env
python main.py -d SUNRGBD -s DEPTH -m FIXMATCH -n_gpu 0 -ns 5 10 25 50 -np 1 2 3 4 5
