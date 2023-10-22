import sys

import torch

sys.path.append("./result/")

import numpy as np
import random
from tqdm import tqdm
from Model.Custom.timestamp import print_time as Timestamp

# Pytorch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

# Dataset
from Data.Dataset import NTUDataSet

# Extension
from Model.Extension.rich import *
from Model.Extension.va import ViewAdaptive

# Model
from model import Teacher_Model

from Model.msg3d_va_rich import Model

# from Model.msg3d import Model
from Model.MS_G3D.Utils.ntu_rgb_d import AdjMatrixGraph
# from MSG3D.model.msg3d import Model, MultiWindow_MS_G3D
from Model.MS_G3D.Components.ms_gcn import MultiScale_GraphConv as MS_GCN
from Model.MS_G3D.Components.ms_tcn import MultiScale_TemporalConv as MS_TCN

# Loss function
from Model.Function.mpjpe import *

#################### * Random Seed Set * ####################
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=800)

#################### * Config parameters * ####################
# Model selection
option = "MS_G3D_VA_RICH"
# [1]: AAGCN
# [2]: MSG3D
# [3]: MS_G3D_VA_RICH

# Training & Testing
num_epochs = 60
train_batch_size = 16  # AAGCN = 12 | MS-G3D = 8 (Experiment)
test_batch_size = 32
learning_rate = 0.05  # AAGCN = 0.1 | MS-G3D = 0.05 (Paper)
frame = 300
downsample_rate = 1  # Default = 1

# Optimizer
momentum = 0.9
weight_decay = 0.0005  # AAGCN = 0.0001 | MS-G3D = 0.0005 (Paper)
nesterov = True
step = [30, 40]

# Path
data_path = "/media/user/DATA/"
result_path = "./result/"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arg_config
args = {
    "dataset_config": {
        "directory": "{}VA+RICH5+MS-G3D_EF/data/ntu/".format(data_path),
        "partition": "xsub",
        "use_mmap": True,
        "debug": False,
        "downsample_rate": 1,
    }
}
args = args["dataset_config"]

# train_dataset_config
train_dataset_config = {
    "data_path": args["directory"] + args["partition"] + "/train_multi_joint_60.npy",
    "label_path": args["directory"] + args["partition"] + "/train_label_60.pkl",
    "use_mmap": args["use_mmap"],
    "debug": args["debug"],
    "downsample_rate": args["downsample_rate"],
}

# val_dataset_config
val_dataset_config = {
    "data_path": args["directory"] + args["partition"] + "/val_multi_joint_60.npy",
    "label_path": args["directory"] + args["partition"] + "/val_label_60.pkl",
    "use_mmap": args["use_mmap"],
    "debug": args["debug"],
}

#################### * Dataset config * ####################
# Training
train_dataset = NTUDataSet(train_dataset_config)

# # Not using downsample
if downsample_rate == 1:
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )

# Using downsample
else:
    randomSampler_train = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=int(len(train_dataset) / downsample_rate),
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=randomSampler_train,
    )

# Testing
test_dataset = NTUDataSet(val_dataset_config)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

#################### * Teacher model * ####################

if option == "MS_G3D_VA_RICH":
    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
    )



# model = Model_teacher(
#     num_class=60,
#     num_point=25,
#     num_person=2,
#     num_gcn_scales=13,
#     num_g3d_scales=6,
#     in_channels=3,
#     ATU_layer=2,
#     T=300,
#     option="MSG3D",  # AAGCN_VA_RICH / MS_G3D_VA_RICH
#     # option="AAGCN_VA_RICH",
# )


#################### * Optimizer * ####################
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=nesterov,
)

#################### * Training model * ####################
# if option == "MS_G3D_VA_RICH":
#     # Using pretrained model
#     path = "./MSG3D/pretrained-models/ntu60-xsub-joint-paper.pt"
#     model.load_state_dict(torch.load(path))

#     # Define parameters
#     Graph = AdjMatrixGraph
#     A_binary = Graph().A_binary
#     num_g3d_scales = 6
#     num_gcn_scales = 13
#     c1 = 96

#     # Adding VA
#     model.va = ViewAdaptive()

#     # Adding RICH (Modify parameters for RICH)
#     model.data_bn = nn.BatchNorm1d(750)
#     in_channels = 15

#     model.gcn3d1 = MultiWindow_MS_G3D(
#         in_channels, c1, A_binary, num_g3d_scales, window_stride=1
#     )
#     model.sgcn1 = nn.Sequential(
#         MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
#         MS_TCN(c1, c1),
#         MS_TCN(c1, c1),
#     )

model.to(device)
model.train()

# #################### * Loss function * ####################
criterion_crossentropy = nn.CrossEntropyLoss()

# Selection the best epoch for testing
lowest_loss = float("inf")

for epoch in range(num_epochs):
    # Initial value
    total_loss = 0.0

    # Training epochs
    for idata, data in enumerate(train_dataloader):
        idx = data["idx"].to(device)
        label = data["label"].to(device)
        feature = data["data"][:, :, :frame, :, :].to(device)

        predict_action_class = model(feature)
        loss = criterion_crossentropy(predict_action_class, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predict_label = torch.max(predict_action_class, 1)[1]
            acc = torch.mean((predict_label == label).float())
            training_loss = loss.item()

            # Print every 50 steps
            if idata % 50 == 0:
                print(
                    "epoch {} | step: {} | Training_loss: {:.8f} | Training_Acc: {:.8f}".format(
                        epoch, idata, training_loss, acc
                    )
                )

    # Testing per epoch
    for data in tqdm(test_dataloader):
        idx = data["idx"].to(device)
        label = data["label"].to(device)
        feature = data["data"].to(device)[:, :, :frame, :, :]

        with torch.no_grad():
            predict_action_class = model(feature)
            loss = criterion_crossentropy(predict_action_class, label)
            predict_label = torch.max(predict_action_class, 1)[1]
            acc += torch.mean((predict_label == label).float())
            total_loss += loss.item()

    print(
        "epoch {} | Testing_loss: {} | Testing_Acc: {}".format(
            epoch,
            total_loss / len(test_dataloader),
            acc / len(test_dataloader),
        )
    )

    # Save training log
    f = open("{}{}/{}_Train_teacher_log".format(result_path, option, option), "a")
    f.write(
        Timestamp(),
        "epoch {} | step: {} | Training_loss: {:.8f} | Training_Acc: {:.8f}".format(
            epoch, idata, training_loss, acc
        ),
    )
    f.write(
        Timestamp(),
        "epoch {} | Testing_loss: {} | Testing_Acc: {}".format(
            epoch,
            total_loss / len(test_dataloader),
            acc / len(test_dataloader),
        ),
    )
    f.close()

    epoch_loss = total_loss / len(test_dataset)

    #################### * Save model * ####################
    # Best result
    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        torch.save(
            model.state_dict(),
            "{}{}/{}_pre_best_epoch{}.pth".format(result_path, option, option, epoch),
        )

    # Checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            "{}{}/{}_pre_epoch{}.pth".format(result_path, option, option, epoch),
        )

    # Last result
    torch.save(
        model.state_dict(),
        "{}{}/{}_pre_last_epoch.pth".format(result_path, option, option),
    )


#################### * Test model * ####################
# Best result only for testing
model.load_state_dict(
    torch.load(
        "{}{}/{}_pre_best_epoch{}.pth".format(result_path, option, option, epoch)
    )
)


model.to(device)
model.eval()

# Initial value
acc = 0.0

for data in tqdm(test_dataloader):
    label = data["label"].to(device)
    feature = data["data"].to(device)[:, :, :frame, :, :]

    with torch.no_grad():
        predict_action_class = model(feature)
        predict_label = torch.max(predict_action_class, 1)[1]
        acc += torch.mean((predict_label == label).float())

    # Save testing log
    f = open("{}{}/{}_Test_teacher_log".format(result_path, option, option), "a")
    f.write(Timestamp(), "Testing_Acc: {}".format(acc) + "\n")
    f.close()


print(
    Timestamp(),
    "Testing_Acc: {}".format(
        acc / len(test_dataloader),
    ),
)

print(Timestamp(), "#################### Teacher Model Complete ####################")
