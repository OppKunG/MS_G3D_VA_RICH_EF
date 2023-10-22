import sys

sys.path.append("./result/")

import numpy as np
import random
from tqdm import tqdm

# Pytorch
import torch
from torch.utils.data import DataLoader

# Dataset
from Data.Dataset import NTUDataSet

# Feature
from Model.Custom.timestamp import print_time as Timestamp

# AAGCN
from Src.AAGCN.GCN_Unsupervised.unsupervised_globalgcn_3layer_action_KD_atu import (
    Model_student as AAGCN_Model_student,
)

# MS-G3D
from model import Training_Student_Model as Training_Student_Model

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
# [3]: MSG3D_VA_RICH

# Training & Testing
num_epochs = 60
train_batch_size = 12  # AAGCN = 16 | MS-G3D = 12 (Experiment)
test_batch_size = 32
frame = 300

# Path
data_path = "/media/user/DATA/VA+RICH5+MS-G3D_EF/"
epoch_path = "./Batch/"
log_path = "./Log/"
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

# val_dataset_config
val_dataset_config = {
    "data_path": args["directory"] + args["partition"] + "/val_multi_joint_60.npy",
    "label_path": args["directory"] + args["partition"] + "/val_label_60.pkl",
    "use_mmap": args["use_mmap"],
    "debug": args["debug"],
}


#################### * Dataset config * ####################
# Testing
test_dataset = NTUDataSet(val_dataset_config)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

#################### * Model * ####################

model_student = Training_Student_Model(
    num_class=60,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    in_channels=3,
    ATU_layer=2,
    adaptive=True,
    attention=True,
)

#################### * Testing model * ####################
num_best_epoch = 60
lowest_loss = 0.06364738941192627
print("Best epoch: {}, Lowest loss: {}".format(num_best_epoch, lowest_loss))
model_student.load_state_dict(
    torch.load("{}{}_post_best_epoch{}.pth".format(epoch_path, option, num_best_epoch))
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
                num_best_epoch, lowest_loss, acc_student, predict_label_student, label
            )
        )
        f.close()

        total_acc_student += acc_student.item()

total_acc_student = total_acc_student / len(test_dataloader.dataset) * test_batch_size


#################### * Output * ####################
print("Best epoch: {}, Lowest loss: {:.4f} \n".format(num_best_epoch, lowest_loss))
print("Testing_Student_Acc: {}".format(total_acc_student))

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
    "#################### Testing Action Recognition Model Complete ####################",
)
