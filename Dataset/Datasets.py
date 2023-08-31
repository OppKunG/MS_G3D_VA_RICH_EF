import numpy as np
import pickle
import random

import torch
import torch.utils.data as data

from DLNest.Common.DatasetBase import DatasetBase

class NTUDataSet(data.Dataset):
    def __init__(self,arg:dict):
       
        with open(arg['label_path'],'rb') as f:
            self.sample_name, self.label = pickle.load(f, encoding='latin1')
        # np.load https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.load.html

        if arg['use_mmap']:
            self.data = np.load(arg['data_path'], mmap_mode='r')
        else:
            self.data = np.load(arg['data_path'])
        if arg['debug']:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]


    def __len__(self):
        return (len(self.label))

    def __getitem__(self,idx):
        # np.array和 np.asarray的区别 https://www.cnblogs.com/keye/p/11264599.html
        data = np.array(self.data[idx])
        label = self.label[idx]
        sample = dict(data=data,label=label,idx=idx)   
        return sample

class Dataset(DatasetBase):
    def __init__(self,args : dict):
        self.args = args["dataset_config"]
        train_dataset_config = {
            # "data_path":self.args["directory"]+self.args["partition"]+"/train_data_joint.npy",#change
            "data_path":self.args["directory"]+self.args["partition"]+"/train_multi_joint.npy",
            "label_path":self.args["directory"]+self.args["partition"]+"/train_label.pkl",
            "use_mmap":self.args["use_mmap"],
            "debug":self.args["debug"],
            "downsample_rate":args["downsample_rate"]
        }
        val_dataset_config = {
            # "data_path":self.args["directory"]+self.args["partition"]+"/val_data_joint.npy",#change
            "data_path":self.args["directory"]+self.args["partition"]+"/val_multi_joint.npy",
            "label_path":self.args["directory"]+self.args["partition"]+"/val_label.pkl",
            "use_mmap":self.args["use_mmap"],
            "debug":self.args["debug"],
            "downsample_rate":args["downsample_rate"]
        }

        self.trainSet = NTUDataSet(train_dataset_config)
        self.valSet = NTUDataSet(val_dataset_config)

        def init_seed(_):
            torch.cuda.manual_seed_all(1)
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            # torch.backends.cudnn.enabled = False
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        if args["is_test"]:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=False,
                num_workers=self.args['num_workers'],
                drop_last=False,
                worker_init_fn=init_seed
                )
        else:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=True,
                num_workers=self.args['num_workers'],
                drop_last=False,
                worker_init_fn=init_seed
                )
        self.valLoader = torch.utils.data.DataLoader(
            dataset=self.valSet,
            batch_size=self.args['batchsize'],
            shuffle=False,
            num_workers=self.args['num_workers'],
            drop_last=False,
            worker_init_fn=init_seed
        )
        
    def afterInit(self):
        return {},self.trainLoader,self.valLoader