import sys

sys.path.append("./result/")

import numpy as np
import random
from tqdm import tqdm
from Model.Custom.timestamp import print_time as Timestamp

# Pytorch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

# Dataset
from Data.Dataset import NTUDataSet

from Model.Extension.rich import *
from Model.Extension.va import ViewAdaptive

# AAGCN
from Src.AAGCN.GCN_Unsupervised.unsupervised_globalgcn_3layer_action_KD_atu import (
    Model_student as AAGCN_Model_student,
    Model as AAGCN_VA_RICH,
)


# MS-G3D
from model import Training_Student_Model as Training_Student_Model

# from Model.msg3d_va_rich import Model as MSG3D_VA_RICH
from Model.MS_G3D.Utils.ntu_rgb_d import AdjMatrixGraph
from Model.MS_G3D.msg3d import Model, MultiWindow_MS_G3D
from Model.MS_G3D.Components.ms_gcn import MultiScale_GraphConv as MS_GCN
from Model.MS_G3D.Components.ms_tcn import MultiScale_TemporalConv as MS_TCN

# Loss function
from Model.Function.mpjpe import *

import torch.backends.cudnn as cudnn

# speed up
cudnn.fastest = True
cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
torch.cuda.empty_cache()


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
# [3]: MSG3D_VA_RICH


# Training & Testing
num_epochs = 60
train_batch_size = 12  # AAGCN = 16 | MS-G3D = 12 (Experiment)
test_batch_size = 32
learning_rate = 0.05  # AAGCN = 0.1 | MS-G3D = 0.05
frame = 300
downsample_rate = 1  # Default = 1

# Optimizer
momentum = 0.9
weight_decay = 0.0005  # AAGCN = 0.0001 | MS-G3D = 0.0005
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
        "partition": "xsub/",
        "use_mmap": True,
        "batchsize": 16,
        "num_workers": 4,
        "debug": False,
        "occlusion_part": [4],  # choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # 1:left arm,
        # 2:right arm
        # 3:two hands
        # 4:two legs
        # 5:trunk
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
    # print(len(train_dataloader))
    # print(len(train_dataloader.dataset))
    # print(train_batch_size)

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

#################### * Model * ####################

# AAGCN
# model_student = AAGCN_Model_student(
#     num_class=60,
#     num_point=25,
#     num_person=2,
#     graph=A,
#     graph_args=dict(),
#     ATU_layer=2,
#     in_channels=3,
#     if_rotation=False,
#     seg_num=1,
#     if_vibrate=False,
#     prediction_mask=0,
#     GCNEncoder="AGCN",
#     T=300,
#     predict_seg=1,
# )

# model_teacher = AAGCN_Model(
#     num_class=60, num_point=25, num_person=2, graph=A, graph_args=dict()
# )

# MSG3D

model_student = Training_Student_Model(
    num_class=60,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    in_channels=3,
    ATU_layer=2,
    T=300,
    option="MSG3D_VA_RICH",  # AAGCN / MSG3D
)

# Teacher
model_teacher = Model(
    num_class=60,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
)

# Define parameters
Graph = AdjMatrixGraph
A_binary = Graph().A_binary
num_g3d_scales = 6
num_gcn_scales = 13
c1 = 96

# Adding VA
model_teacher.va = ViewAdaptive()

# Adding RICH (Modify parameters for RICH)
model_teacher.data_bn = nn.BatchNorm1d(750)
in_channels = 15

model_teacher.gcn3d1 = MultiWindow_MS_G3D(
    in_channels, c1, A_binary, num_g3d_scales, window_stride=1
)
model_teacher.sgcn1 = nn.Sequential(
    MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
    MS_TCN(c1, c1),
    MS_TCN(c1, c1),
)


#################### * Optimizer * ####################
optimizer = torch.optim.SGD(
    model_student.model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=nesterov,
)


#################### * Training model * ####################

model_teacher.load_state_dict(torch.load("./MS_G3D_VA_RICH_pre_best_epoch.pth"))


model_teacher.to(device)
model_student.to(device)

with torch.no_grad():
    model_teacher.eval()
model_student.train()


#################### * Loss function * ####################

# Selection the best epoch for testing
lowest_loss = float("inf")

criterion_mse = nn.MSELoss()
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_kld = nn.KLDivLoss(reduction="batchmean")

for epoch in range(num_epochs):
    # Initial value
    total_loss = 0.0
    total_error = 0.0
    acc_teacher = 0.0
    total_acc_student = 0.0
    total_mpjpe = 0.0
    total_mse = 0.0
    total_crossentropy = 0.0
    total_kld = 0.0

    for data in tqdm(train_dataloader):
        label = data["label"].to(device)
        data = data["data"].to(device)
        data = data[:, :, :frame, :, :]
        feature = data.clone()

        # Testing Teacher model accuracy
        with torch.no_grad():
            predict_action_teacher = model_teacher(feature)
            predict_label_teacher = torch.max(predict_action_teacher, 1)[1]
            acc_teacher += torch.mean((predict_label_teacher == label).float())

        reconstructed = model_student(feature)
        predict_action_student = reconstructed[2]
        x_mask = reconstructed[3]
        reconstructed = reconstructed[0] * (1 - x_mask) + data * (x_mask)

        error_orig = mpjpe(data, data, reverse_center=False)
        error_recon = mpjpe(reconstructed, data)

        # Total error
        total_error += (error_recon - error_orig).item()
        total_mpjpe += total_error
        # total_error = total_error / len(train_dataloader)

        predict_label_student = torch.max(predict_action_student, 1)[1]
        acc_student = torch.mean((predict_label_student == label).float())

        total_acc_student += acc_student.item()
        loss_mse = criterion_mse(reconstructed, data)
        loss_crossentropy = criterion_crossentropy(predict_action_student, label)
        loss_kld = criterion_kld(
            F.log_softmax(predict_action_student, dim=1),
            F.softmax(predict_action_teacher, dim=1),
        )

        loss = loss_kld + loss_crossentropy

        f = open("{}{}/xx".format(result_path, option), "a")
        f.write("\n" + str(loss) + "\n" + str(loss_crossentropy) + str(loss_kld))
        f.close()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Total loss
        total_loss += loss.detach()
        total_mse += loss_mse.detach()
        total_crossentropy += loss_crossentropy.detach()
        total_kld += loss_kld.detach()

    total_mpjpe = total_mpjpe / len(train_dataloader)
    total_mse = total_mse / len(train_dataloader)
    total_crossentropy = total_crossentropy / len(train_dataloader)
    total_kld = total_kld / len(train_dataloader)

    print("total_mpjpe", total_mpjpe)
    print("total_mse", total_mse)
    print("total_crossentropy", total_crossentropy)

    epoch_loss = total_loss / len(train_dataset)
    error = mpjpe(reconstructed, data)

    #################### * Save model state * ####################

    # Best result
    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        torch.save(
            model_student.state_dict(),
            "{}{}/{}_post_best_epoch{}.pth".format(result_path, option, option, epoch),
        )

    # Checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save(
            model_student.state_dict(),
            "{}{}/{}_post_epoch{}.pth".format(result_path, option, option, epoch),
        )

    # Last result
    torch.save(
        model_student.state_dict(),
        "{}{}/{}_post_last_epoch.pth".format(result_path, option, option),
    )

    print("total_acc_teacher:", acc_teacher / len(train_dataloader))
    print("total_acc_student:", total_acc_student / len(train_dataloader))
    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, epoch_loss))


#################### * Load model state * ####################
model_student.load_state_dict(
    torch.load("{}{}/{}_post_last_epoch.pth".format(result_path, option, option))
)
model_student.to(device)
model_student.eval()

with torch.no_grad():
    total_error = 0.0
    total_error_all = 0.0
    total_mpjpe = 0.0
    total_mpjpe_all = 0.0
    total_acc_student = 0.0
    tqdm_time = 0

    for data in tqdm(test_dataloader):
        tqdm_time += 1
        label = data["label"].to(device)
        data = data["data"].to(device)
        data = data[:, :, :frame, :, :]

        feature = data.clone()
        reconstructed = model_student(feature)
        predict_action_student = reconstructed[2]
        x_mask = reconstructed[3]

        feature = feature * x_mask
        reconstructed_all = reconstructed[0]
        reconstructed = reconstructed[0] * (1 - x_mask) + data * (x_mask)

        error_recon_all = mpjpe(reconstructed_all, data)
        error_recon = mpjpe(reconstructed, data)

        total_error_all += error_recon_all.item()
        total_error += error_recon.item()
        total_mpjpe_all = total_error_all / tqdm_time
        total_mpjpe = total_error / tqdm_time
        predict_label_student = torch.max(predict_action_student, 1)[1]
        acc_student = torch.mean((predict_label_student == label).float())

        f = open("{}xxxxxxxxxx_test".format(result_path), "a")
        f.write(str(acc_student) + str(predict_label_student) + str(label) + "\n")
        f.close()
        total_acc_student += acc_student.item()

total_acc_student = total_acc_student / len(test_dataloader)


#################### * Output * ####################
print("total_acc", total_acc_student)
# print('total_error',total_error)
print("total_mpjpe_all", total_mpjpe_all)
print("total_mpjpe", total_mpjpe)
print("data.shape: ", data.shape)

# video_number
video_number = data.shape[0]
print("video_number: ", video_number)
print("label", label, label.shape)
print("predict_label", predict_label_student, predict_label_student.shape)

# label
label = label
np.save("{}label".format(result_path), label.cpu().detach().numpy())

# predict_label
predict_label_student = predict_label_student
np.save("{}predict_label".format(result_path), predict_label_student.cpu().detach().numpy())

# 3d_pose_ori
data = data[:, :, :, :, :]
data = (
    data.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, data.shape[4])
)
np.save("{}3d_pose_ori".format(result_path), data.cpu().detach().numpy())

# data
f = open("{}data".format(result_path), "w")
f.write(str(data.cpu().detach().numpy()))
f.close()

# 3d_pose0
feature = feature[:, :, :, :, :]
feature = (
    feature.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, data.shape[4])
)
np.save("{}3d_pose0".format(result_path), feature.cpu().detach().numpy())

# 3d_pose
reconstructed = reconstructed[:, :, :, :, :]
reconstructed = (
    reconstructed.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, data.shape[4])
)
np.save("{}3d_pose".format(result_path), reconstructed.cpu().detach().numpy())

# reconstructed
f = open("{}reconstructed".format(result_path), "w")
f.write(str(reconstructed.cpu().detach().numpy()))
f.close()

print(
    Timestamp(),
    "#################### Action Recognition Model Complete ####################",
)
