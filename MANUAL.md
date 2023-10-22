# Installation environment in conda
- conda create -n NAME_PROJECT python
- conda activate NAME_PROJECT
- python -m pip install -r requirements.txt

# How to run
- cd data_gen

## Generate data from raw data
- python ntu_gendata.py

## Running all my work
- python main.py

# Monitoring GPU CMD:
- watch -n 0.1 nvidia-smi

# If you want to running some file only

## Pre-trained model teacher
- python train_action_pre.py 

## Training model
- python train_action_post.py

## Testing model
- python test_action_post.py

## Convert npy file to pictures
- python vis_pose.py

