import sys

sys.path.append("./result/")

import numpy as np
import random
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

# Dataset
from Data.Dataset import NTUDataSet

# Feature
from Model.Custom.timestamp import print_time as Timestamp

# AAGCN
from Src.AAGCN.GCN_Unsupervised.unsupervised_globalgcn_3layer_action_KD_atu import (
    Model_student as AAGCN_Model_student,
    Model as AAGCN_VA_RICH,
)

# MS-G3D
from model import Training_Student_Model
from model import Testing_Student_Model
from Model.msg3d_va_rich import Model as MS_G3D_VA_RICH

# Loss function
from Model.Function.mpjpe import *


#################### * Random Seed Set * ####################
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=800)


#################### * Config parameters * ####################
# Model selection
model_name = "MS_G3D_VA_RICH"  # AAGCN_VA_RICH | MS_G3D | MSG3D_VA_RICH
print("Model Name: {}".format(model_name))

# Training & Testing
num_epochs = 60
print("Number of epochs: {}".format(num_epochs))

train_batch_size = 12  # AAGCN = 16 | MS-G3D = 12 (Experiment)
test_batch_size = 32
frame = 300
downsample_rate = 1  # Default = 1 | Testing code (Debug only) = 100 (10 sec/epoch)

# Optimizer
learning_rate = 0.05  # AAGCN = 0.1 | MS-G3D = 0.05
momentum = 0.9
weight_decay = 0.0005  # AAGCN = 0.0001 | MS-G3D = 0.0005
nesterov = True

# Path
data_path = "/media/user/DATA/VA+RICH5+MS-G3D_EF/"
save_teacher_path = "./Batch/Teacher_Model/"
epoch_path = "./Batch/Student_Model/"
log_path = "./Log/Student_Model/"
vis_path = "./Archive/"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# arg_config
args = {
    "dataset_config": {
        "directory": "{}data/ntu/".format(data_path),
        "partition": "xsub/",
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
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


#################### * Training model * ####################
model_teacher = MS_G3D_VA_RICH(
    num_class=60,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
)

model_teacher.load_state_dict(
    torch.load(
        "{}{}_[70]_(85.93%)/{}_pre_best_epoch70.pth".format(
            save_teacher_path, model_name, model_name
        )
    )
)  # Best teacher by Jerry

model_teacher.to(device)
model_student = Training_Student_Model(menu=model_name)
model_student.to(device)

with torch.no_grad():
    model_teacher.eval()
model_student.train()

# Optimizer
optimizer = torch.optim.SGD(
    model_student.model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=nesterov,
)

# Initial value
num_best_epoch = 0
lowest_loss = float("inf")
criterion_mse = nn.MSELoss()
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_kld = nn.KLDivLoss(reduction="batchmean")

for epoch in range(num_epochs):
    # Initial value
    total_loss = 0.0
    total_error = 0.0
    total_acc_teacher = 0.0
    total_acc_student = 0.0
    total_mpjpe = 0.0
    total_mse = 0.0
    total_crossentropy = 0.0
    total_kld = 0.0
    # loss_value = []

    for data in tqdm(train_dataloader):
        label = data["label"].to(device)
        feature = data["data"].to(device)[:, :, :frame, :, :]

        # Testing Teacher model
        with torch.no_grad():
            predict_action_teacher = model_teacher(feature)
            predict_label_teacher = torch.max(predict_action_teacher, 1)[1]

        reconstructed = model_student(feature)
        predict_action_student = reconstructed[1]
        x_mask = reconstructed[2]
        reconstructed = reconstructed[0] * (1 - x_mask) + feature * (x_mask)

        error_orig = mpjpe(feature, feature)
        error_recon = mpjpe(reconstructed, feature)

        # Total error
        total_error += (error_recon - error_orig).item()
        total_mpjpe += total_error
        total_error = total_error / len(train_dataloader.dataset)

        #
        predict_label_student = torch.max(predict_action_student, 1)[1]
        acc_teacher = torch.mean((predict_label_teacher == label).float())
        acc_student = torch.mean((predict_label_student == label).float())

        total_acc_teacher += acc_teacher.item()
        total_acc_student += acc_student.item()
        loss_mse = criterion_mse(reconstructed, feature)
        loss_crossentropy = criterion_crossentropy(predict_action_student, label)
        loss_kld = criterion_kld(
            F.log_softmax(predict_action_student, dim=1),
            F.softmax(predict_action_teacher, dim=1),
        )

        loss = loss_kld + loss_crossentropy

        # loss_value.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Total loss
        total_loss += loss.detach()
        total_mse += loss_mse.detach()
        total_crossentropy += loss_crossentropy.detach()
        total_kld += loss_kld.detach()

    print("Total_MPJPE: {}".format(total_mpjpe / len(train_dataloader.dataset)))
    print("Total_MSE: {}".format(total_mse / len(train_dataloader.dataset)))
    print(
        "Total_Crossentropy: {}".format(
            total_crossentropy / len(train_dataloader.dataset)
        )
    )
    print("Total_KLD: {}".format(total_kld / len(train_dataloader.dataset)))

    epoch_loss = total_loss / len(train_dataset)
    error = mpjpe(reconstructed, feature)

    #################### * Saving model * ####################

    # Best epoch
    if epoch_loss < lowest_loss:
        num_best_epoch = epoch + 1
        lowest_loss = epoch_loss
        torch.save(
            model_student.state_dict(),
            "{}{}_post_best_epoch{}.pth".format(epoch_path, model_name, epoch + 1),
        )

    # Checkpoint epochs every 10
    if (epoch + 1) % 10 == 0:
        torch.save(
            model_student.state_dict(),
            "{}{}_post_epoch{}.pth".format(epoch_path, model_name, epoch + 1),
        )

    # Last epoch
    torch.save(
        model_student.state_dict(),
        "{}{}_post_last_epoch.pth".format(epoch_path, model_name),
    )

    # Output
    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, epoch_loss))
    print(
        "Teacher_Acc: {}".format(
            total_acc_teacher
            / len(train_dataloader.dataset)
            * train_batch_size
            * downsample_rate
        )
    )
    print(
        "Student_Acc: {}".format(
            total_acc_student
            / len(train_dataloader.dataset)
            * train_batch_size
            * downsample_rate
        )
    )

    # Training log file
    f = open("{}Train_Student_log.txt".format(log_path), "a")
    f.write(
        "{} Epoch: {}, Loss: {:.4f}, Crossentropy_loss: {:.4f}, KLD_loss: {:.4f} \n Predict_label: {} \n Label: {} \n".format(
            Timestamp(),
            epoch + 1,
            epoch_loss,
            loss_crossentropy,
            loss_kld,
            predict_action_student,
            label,
        )
    )
    f.close()


#################### * Testing model * ####################
print("Best epoch: {}, Lowest loss: {}".format(num_best_epoch, lowest_loss))

for occlusion_part in range(1, 13):
    model_student = Testing_Student_Model(menu=model_name, part=occlusion_part)
    model_student.load_state_dict(
        torch.load(
            "{}{}_post_best_epoch{}.pth".format(epoch_path, model_name, num_best_epoch)
        )
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
            feature = data["data"].to(device)[:, :, :frame, :, :]

            mask_data = feature.clone()

            reconstructed = model_student(feature)
            predict_action_student = reconstructed[2]
            x_mask = reconstructed[3]

            mask_data = mask_data * x_mask
            reconstructed_all = reconstructed[0]
            reconstructed = reconstructed[0] * (1 - x_mask) + feature * (x_mask)

            error_recon_all = mpjpe(reconstructed_all, feature)
            error_recon = mpjpe(reconstructed, feature)

            total_error_all += error_recon_all.item()
            total_error += error_recon.item()
            total_mpjpe_all = total_error_all / tqdm_time
            total_mpjpe = total_error / tqdm_time

            predict_label_student = torch.max(predict_action_student, 1)[1]
            acc_student = torch.mean((predict_label_student == label).float())

            # Testing log file
            f = open("{}Test_Student_log.txt".format(log_path), "a")
            f.write(
                "Best epoch: {}, Lowest loss: {:.4f}, Acc: {} \n Predict_label: {} \n Label: {} \n".format(
                    num_best_epoch,
                    lowest_loss,
                    acc_student,
                    predict_label_student,
                    label,
                )
            )
            f.close()

            total_acc_student += acc_student.item()

    total_acc_student = (
        total_acc_student / len(test_dataloader.dataset) * test_batch_size
    )

    #################### * Output * ####################
    print("Best epoch: {}, Lowest loss: {:.4f} \n".format(num_best_epoch, lowest_loss))
    print(
        "Occlution parts: {}, Testing_Student_Acc: {}".format(
            occlusion_part, total_acc_student
        )
    )

    print("total_mpjpe_all", total_mpjpe_all)
    print("total_mpjpe", total_mpjpe)
    print("feature.shape: ", feature.shape)

# video_number
video_number = feature.shape[0]
print("video_number: ", video_number)
print("label", label, label.shape)
print("predict_label", predict_label_student, predict_label_student.shape)

# label
label = label
np.save("{}label".format(vis_path), label.cpu().detach().numpy())

# predict_label
predict_label_student = predict_label_student
np.save(
    "{}predict_label".format(vis_path), predict_label_student.cpu().detach().numpy()
)

# 3d_pose_ori
feature = feature[:, :, :, :, :]
feature = (
    feature.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, feature.shape[4])
)
np.save("{}3d_pose_ori".format(vis_path), feature.cpu().detach().numpy())

# data
f = open("{}data".format(vis_path), "w")
f.write(str(feature.cpu().detach().numpy()))
f.close()

# 3d_pose0
mask_data = mask_data[:, :, :, :, :]
mask_data = (
    mask_data.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, feature.shape[4])
)
np.save("{}3d_pose0".format(vis_path), mask_data.cpu().detach().numpy())

# 3d_pose
reconstructed = reconstructed[:, :, :, :, :]
reconstructed = (
    reconstructed.permute(0, 2, 3, 1, 4)
    .contiguous()
    .view(video_number, frame, 25, 3, feature.shape[4])
)
np.save("{}3d_pose".format(vis_path), reconstructed.cpu().detach().numpy())

# reconstructed
f = open("{}reconstructed".format(vis_path), "w")
f.write(str(reconstructed.cpu().detach().numpy()))
f.close()

print(
    Timestamp(),
    "#################### Action Recognition Model Complete ####################",
)
