# CCU_Internship
**Goal:** VA+RICH5+MS-G3D_EF (1s)

*Conda Technique*
```
conda env list
conda create -n NAME_PROJECT python==3.8 (version) 
conda activate NAME_PROJECT
conda deactivate

conda remove -n NAME_PROJECT --all 

pip freeze --all > requirements.txt

python -m pip install -r requirements.txt
```


1. git clone
2. conda create
3. install requirements
4. Add directory /data
5. Unzip data
6. Open cmd/termminal
7. cd data_gen
8. python ntu_gendata.py
9. python train_action_pre_atu.py
10. 
