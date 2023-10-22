import sys

sys.path.append("./result/")

import numpy as np
import random
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

# Dataset
from Data.Dataset import NTUDataSet

# Feature
from Model.Custom.timestamp import print_time as Timestamp

# Model
from model import Teacher_Model

# from AAGCN.GCN_Unsupervised.unsupervised_globalgcn_3layer_action_KD_atu import (
#     Model_teacher,
# )  # change 3layer+action


#################### * Random Seed Set * ####################
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=800)


#################### * Config parameters * ####################
# Model
model_name = "MS_G3D_VA_RICH"  # AAGCN_VA_RICH | MS_G3D | MS_G3D_VA_RICH
print("Model Name: {}".format(model_name))

# Training & Testing
num_epochs = 70  # AAGCN = 60 | MS-G3D = 70 (Experiment)
print("Number of epochs: {}".format(num_epochs))

train_batch_size = 16  # AAGCN = 12 | MS-G3D = 16 (Experiment)
test_batch_size = 32
frame = 300
downsample_rate = 1  # Default = 1 | Testing code (Debug only) = 100 (10 sec/epoch)

# Optimizer
learning_rate = 0.05  # AAGCN = 0.1 | MS-G3D = 0.05 (Paper)
momentum = 0.9
weight_decay = 0.0005  # AAGCN = 0.0001 | MS-G3D = 0.0005 (Paper)
nesterov = True

# Path
data_path = "/media/user/DATA/VA+RICH5+MS-G3D_EF/"
epoch_path = "./Batch/Teacher_Model/"
log_path = "./Log/Teacher_Model/"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arg_config
args = {
    "dataset_config": {
        "directory": "{}data/ntu/".format(data_path),
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
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


#################### * Training model * ####################
model = Teacher_Model(menu=model_name)
model.to(device)
model.train()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=nesterov,
)


# Initial value
num_best_epoch = 0
lowest_loss = float("inf")
criterion_crossentropy = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    # Initial value
    total_loss = 0.0

    for data in tqdm(train_dataloader):
        label = data["label"].to(device)  # [16, 3, 300, 25, 2]
        feature = data["data"].to(device)[:, :, :frame, :, :]

        predict_action_class = model(feature)
        loss = criterion_crossentropy(predict_action_class, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_dataset)

    #################### * Saving model * ####################

    # Best epoch
    if epoch_loss < lowest_loss:
        num_best_epoch = epoch + 1
        lowest_loss = epoch_loss
        torch.save(
            model.state_dict(),
            "{}{}_pre_best_epoch{}.pth".format(epoch_path, model_name, epoch + 1),
        ),

    # Checkpoint epochs every 10
    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(),
            "{}{}_pre_epoch{}.pth".format(epoch_path, model_name, epoch + 1),
        )

    # Last epoch
    torch.save(
        model.state_dict(),
        "{}{}_pre_last_epoch.pth".format(epoch_path, model_name),
    )

    # Output
    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, epoch_loss))

    # Training Log file
    f = open("{}Train_Teacher_log.txt".format(log_path), "a")
    f.write(
        "{} Epoch: {}, Loss: {:.4f} \n Predict_label: {} \n Label: {} \n".format(
            Timestamp(), epoch + 1, epoch_loss, predict_action_class, label
        )
    )
    f.close()


#################### * Testing model * ####################
print("Best epoch: {}, Lowest loss: {}".format(num_best_epoch, lowest_loss))
model.load_state_dict(
    torch.load(
        "{}{}_pre_best_epoch{}.pth".format(epoch_path, model_name, num_best_epoch),
    )
)

model.to(device)
model.eval()

# Initial value
total_acc = 0.0

for data in tqdm(test_dataloader):
    label = data["label"].to(device)
    feature = data["data"].to(device)[:, :, :frame, :, :]

    with torch.no_grad():
        predict_action_class = model(feature)
        predict_label = torch.max(predict_action_class, 1)[1]
        acc = torch.mean((predict_label == label).float())

        # Testing log file
        f = open("{}Test_Teacher_log.txt".format(log_path), "a")
        f.write(
            "Best epoch: {}, Lowest loss: {:.4f}, Acc: {} \n Predict_label: {} \n Label: {} \n".format(
                num_best_epoch, lowest_loss, acc, predict_label, label
            )
        )
        f.close()

        total_acc += acc.item()

total_acc = total_acc / len(test_dataloader.dataset) * test_batch_size


#################### * Output * ####################
print("Best epoch: {}, Lowest loss: {:.4f} \n".format(num_best_epoch, lowest_loss))
print("Testing_Teacher_Acc: {}".format(total_acc))

print(
    Timestamp(),
    "#################### Pre-train Teacher Model Complete ####################",
)
