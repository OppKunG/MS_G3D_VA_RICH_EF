import numpy as np
import random

# Pytorch
import torch
import torch.nn as nn

# SRN
from Model.SRN.Components.skeleton import *
from Model.SRN.Components.fc import FC

# from Model.SRN.srn import IPN

# IPN Model
from Src.AAGCN.GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu import (
    AGCNEncoder as SRN,
    AAGCN_VA_RICH,
)

# MS-G3D
from Model.MS_G3D.msg3d import Model as MS_G3D
from Model.msg3d_va_rich import Model as MS_G3D_VA_RICH

random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################### * Pre-trained Model * ####################


class Teacher_Model(nn.Module):
    def __init__(
        self,
        menu,
    ):
        super(Teacher_Model, self).__init__()

        # Models
        self.model_name = menu

        if self.model_name == "AAGCN_VA_RICH":
            self.model = AAGCN_VA_RICH()

        if self.model_name == "MS_G3D":
            self.model = MS_G3D(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

        if self.model_name == "MS_G3D_VA_RICH":
            self.model = MS_G3D_VA_RICH(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

    def forward(self, x):
        return self.model(x)


#################### * Action Recognition * ####################


class Training_Student_Model(nn.Module):
    def __init__(
        self,
        menu,
    ):
        super(Training_Student_Model, self).__init__()

        # SRN
        self.srn = SRN(
            num_class=60,
            num_point=25,
            num_person=2,
            in_channels=3,
            ATU_layer=2,
            adaptive=True,
            attention=True,
        )

        self.full_body = joint_full_body()
        self.occlusion = occlusion_list()
        self.joint = joint_list()
        self.fc = FC()

        # Models
        self.model_name = menu

        if self.model_name == "AAGCN_VA_RICH":
            self.model = AAGCN_VA_RICH()

        if self.model_name == "MS_G3D":
            self.model = MS_G3D(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

        if self.model_name == "MS_G3D_VA_RICH":
            self.model = MS_G3D_VA_RICH(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

    def forward(self, x):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        x_mask = np.ones((x.shape[0], 3, 300, 25, 2))

        #################### * Training occlusion * ####################

        # TODO: None + fix occlusion (1-7) + random miss rate (0.2 0.6)

        num_parts = 13
        # 8 -> none + fix occlusion (1 to 7)
        # 13 -> none + fix occlusion (1 to 7) + random miss rate (0.2 to 0.6)

        part_indices = np.random.choice(num_parts, size=N, replace=True)  # random parts

        for i_b in range(x.shape[0]):
            part_index = part_indices[i_b]
            part = self.occlusion[part_index + 1]

            if part_index <= 8:
                x[i_b, :, :, part, :] = 0
                x_mask[i_b, :, :, part, :] = 0

            else:
                miss_rate = np.random.choice(
                    [0.2, 0.3, 0.4, 0.5, 0.6]
                )  # random miss rate

                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint) * miss_rate)
                    joint_occlusion = np.random.choice(
                        self.joint, joint_occlusion_num, replace=False
                    )
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b, :, t, joint_occlusion, :] = 0

        # GCN_feature
        GCN_feature = torch.empty(N, M, 64, T, V).cuda().to(torch.float32)
        GCN_feature[:, :, :, :, self.full_body - 1] = self.srn(
            x[:, :, :, self.full_body - 1, :]
        )

        # FC
        recover_parts = self.fc(GCN_feature)

        # x_mask
        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(
            device
        )

        # Repaired Skeleton
        repair_skeleton = recover_parts * (1 - x_mask) + ori_data * (x_mask)

        # Model Student Output
        predict_action = self.model(repair_skeleton)

        return recover_parts, predict_action, x_mask


class Testing_Student_Model(nn.Module):
    def __init__(
        self,
        menu,
        part,
    ):
        super(Testing_Student_Model, self).__init__()

        # SRN
        self.srn = SRN(
            num_class=60,
            num_point=25,
            num_person=2,
            in_channels=3,
            ATU_layer=2,
            adaptive=True,
            attention=True,
        )

        self.full_body = joint_full_body()
        self.occlusion = occlusion_list()
        self.joint = joint_list()
        self.fc = FC()

        # Models
        self.model_name = menu

        if self.model_name == "AAGCN_VA_RICH":
            self.model = AAGCN_VA_RICH()

        if self.model_name == "MS_G3D":
            self.model = MS_G3D(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

        if self.model_name == "MS_G3D_VA_RICH":
            self.model = MS_G3D_VA_RICH(
                num_class=60,
                num_point=25,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
            )

        self.part = part

    def forward(self, x):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        x_mask = np.ones((x.shape[0], 3, 300, 25, 2))

        #################### * Testing occlusion * ####################

        if self.part < 8:  # * change occlusion_part
            # TODO: Custom occlusion part (1 to 7)
            x[:, :, :, self.occlusion[self.part], :] = 0
            x_mask[:, :, :, self.occlusion[self.part], :] = 0

        else:
            # TODO: Custom miss rate (0.2 to 0.6)

            miss_rate_choice = 0.2 + (
                (self.part - 8) * 0.1
            )  # * change miss_rate_choice

            for i_b in range(x.shape[0]):
                miss_rate = np.random.choice([miss_rate_choice])
                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint) * miss_rate)
                    joint_occlusion = np.random.choice(
                        self.joint, joint_occlusion_num, replace=False
                    )
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b, :, t, joint_occlusion, :] = 0

        # GCN_feature
        GCN_feature = torch.empty(N, M, 64, T, V).cuda().to(torch.float32)
        GCN_feature[:, :, :, :, self.full_body - 1] = self.srn(
            x[:, :, :, self.full_body - 1, :]
        )

        # FC
        recover_parts = self.fc(GCN_feature)

        # x_mask
        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(
            device
        )

        # Repaired Skeleton
        repair_skeleton = recover_parts * (1 - x_mask) + ori_data * (x_mask)

        # Model Student Output
        predict_action = self.model(repair_skeleton)

        return recover_parts, predict_action, x_mask
