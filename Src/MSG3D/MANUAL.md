# Generating Data
*** NTU RGB+D ***
- cd data_gen
- python ntu_gendata.py
- python gen_bone_data.py --dataset ntu

# Training
- python main.py --config ./config/nturgbd-cross-subject/train_joint.yaml --base-lr 0.05 --device 0 --batch-size 32 --forward-batch-size 16 --work-dir ./results