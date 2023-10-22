# VA+RICH5+MS-G3D_EF (1s)


**Author:** Jakarin Chonchumrus      **Supervisor:** Prof. Wen-Nung Lie

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

```2s-AAGCN Part```
1. git clone
2. conda create
3. install requirements
4. Add directory /data
#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - nturgbd_raw/
    - nturgbd_skeletons_s001_to_s017/     # from `nturgbd_skeletons_s001_to_s017.zip`
    - samples_with_missing_skeletons.txt
```

5. Unzip data
6. Open cmd/termminal
7. cd data_gen
8. python ntu_gendata.py
9. python train_action_pre_atu.py
10. 


```MS-G3D Part```
1. cd project folder
2. git clone https://github.com/NVIDIA/apex
3. cd apex
4. pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

