import numpy as np
import matplotlib.pyplot as plt
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from Model.Custom.timestamp import print_time as Timestamp

vis_path = "./Archive/"
image_path = "./Images/"

# Label of 60 actions
actions = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting position)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
]


def vis_keypoints_3D(vals1, vals2, front):
    ntu60_connections = [
        [0, 1],
        [1, 20],
        [20, 2],
        [2, 3],
        [20, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 21],
        [7, 22],
        [20, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 23],
        [11, 24],
        [0, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [0, 12],
        [12, 13],
        [13, 14],
        [14, 15],
    ]

    ntu60_LR = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        dtype=bool,
    )

    lcolor = "#AA0000"
    rcolor = "#00008B"

    for ind, (i, j) in enumerate(ntu60_connections):
        x1, y1, z1 = [np.array([vals1[i, c], vals1[j, c]]) for c in range(3)]
        x2, y2, z2 = [np.array([vals2[i, c], vals2[j, c]]) for c in range(3)]
        if (ind == 0) or (ind == 1) or (ind == 2) or (ind == 3):
            front.plot(
                x1,
                y1,
                z1,
                alpha=0.8,
                lw=3,
                c="#DAA520",
                marker="o",
                markersize=5,
                markerfacecolor="#DAA520",
            )
            front.plot(
                x2,
                y2,
                z2,
                alpha=0.8,
                lw=3,
                c="#DAA520",
                marker="o",
                markersize=5,
                markerfacecolor="#DAA520",
            )

        else:
            front.plot(
                x1,
                y1,
                z1,
                alpha=0.6,
                lw=3,
                c=lcolor if ntu60_LR[ind] else rcolor,
                marker="o",
                markersize=5,
                markerfacecolor=lcolor if ntu60_LR[ind] else rcolor,
            )
            front.plot(
                x2,
                y2,
                z2,
                alpha=0.6,
                lw=3,
                c=lcolor if ntu60_LR[ind] else rcolor,
                marker="o",
                markersize=5,
                markerfacecolor=lcolor if ntu60_LR[ind] else rcolor,
            )

    # ---------------------------------------------------------------------

    front.set_title("Front View", fontsize=16)

    front.patch.set_facecolor("#FFFFFF")
    front.patch.set_alpha(0.1)

    plt.rcParams["axes.edgecolor"] = "#FFFFFF"
    plt.rc("grid", alpha=0.2, lw=1, linestyle="-", c="#00FFFF")
    front.grid(True)

    front.set_xlim3d([-1000, 1000])
    front.set_ylim3d([-1200, 1200])
    front.set_zlim3d([-1200, 1200])

    front.set_xticks(range(-1000, 1000, 500))
    front.set_yticks(range(-1200, 1200, 600))
    front.set_zticks(range(-1200, 1200, 600))

    plt.setp(front.get_xticklabels(), visible=False)
    plt.setp(front.get_yticklabels(), visible=False)
    plt.setp(front.get_zticklabels(), visible=False)
    front.tick_params(axis="both", which="major", length=0)

    front.xaxis._axinfo["juggled"] = (2, 0, 1)
    front.yaxis._axinfo["juggled"] = (2, 1, 1)
    front.zaxis._axinfo["juggled"] = (2, 2, 2)

    front.view_init(elev=10, azim=90)

    return 0


if __name__ == "__main__":
    fig = plt.figure("3D Skeleton", figsize=(10, 6), dpi=100)

    # Choose number of video [0-6]
    video_number = 3

    for i in range(300):
        plt.ion()

        # RECT can set the position and size of the sub -chart
        rect1 = [
            -0.1,
            0.2,
            0.55,
            0.55,
        ]  # [Left, lower, width, height] prescribed rectangular area
        rect2 = [0.2, 0.2, 0.55, 0.55]
        rect3 = [0.5, 0.2, 0.55, 0.55]

        # Add subgraph ax to Fig, and assign a value position RECT
        ax1 = plt.axes(rect1, projection="3d")
        ax2 = plt.axes(rect2, projection="3d")
        ax3 = plt.axes(rect3, projection="3d")

        # 3d_pose_ori.npy
        output = np.load("{}3d_pose_ori.npy".format(vis_path))
        output11 = output[video_number, :, :, :, 0]
        output12 = output[video_number, :, :, :, 1]
        output11 = np.transpose(output11, (2, 0, 1))
        output12 = np.transpose(output12, (2, 0, 1))

        # 3d_pose0.npy
        output2 = np.load("{}3d_pose0.npy".format(vis_path))
        output21 = output2[video_number, :, :, :, 0]
        output22 = output2[video_number, :, :, :, 1]
        output21 = np.transpose(output21, (2, 0, 1))
        output22 = np.transpose(output22, (2, 0, 1))

        # 3d_pose.npy
        output3 = np.load("{}3d_pose.npy".format(vis_path))
        output31 = output3[video_number, :, :, :, 0]
        output32 = output3[video_number, :, :, :, 1]
        output31 = np.transpose(output31, (2, 0, 1))
        output32 = np.transpose(output32, (2, 0, 1))

        # label.npy
        label = np.load("{}label.npy".format(vis_path))
        label_text = actions[label[video_number]]
        label = label + 1
        label = str(label[video_number])

        # predict_label.npy
        predict_label = np.load("{}predict_label.npy".format(vis_path))
        predict_label_text = actions[predict_label[video_number]]
        predict_label = predict_label + 1
        predict_label = str(predict_label[video_number])

        output11 = output11 * 1000
        output12 = output12 * 1000
        output21 = output21 * 1000
        output22 = output22 * 1000
        output31 = output31 * 1000
        output32 = output32 * 1000

        output11 = torch.Tensor(output11).permute(1, 2, 0)
        output12 = torch.Tensor(output12).permute(1, 2, 0)
        output21 = torch.Tensor(output21).permute(1, 2, 0)
        output22 = torch.Tensor(output22).permute(1, 2, 0)
        output31 = torch.Tensor(output31).permute(1, 2, 0)
        output32 = torch.Tensor(output32).permute(1, 2, 0)

        vis_keypoints_3D(output11[i, :, :], output12[i, :, :], front=ax1)
        vis_keypoints_3D(output21[i, :, :], output22[i, :, :], front=ax2)
        vis_keypoints_3D(output31[i, :, :], output32[i, :, :], front=ax3)

        # Topic
        fig.text(0.06, 0.9, "Real action: ", fontsize=30, ha="left", color="black")
        fig.text(0.9, 0.9, label_text, fontsize=30, ha="right", color="red")
        fig.text(0.03, 0.2, "Predict action: ", fontsize=30, ha="left", color="black")
        fig.text(0.9, 0.2, predict_label_text, fontsize=30, ha="right", color="red")

        # Column title
        fig.text(0.18, 0.81, "Ground truth ", fontsize=16, ha="center", color="black")
        fig.text(0.48, 0.81, "Masked ", fontsize=16, ha="center", color="black")
        fig.text(0.78, 0.81, "Repaired ", fontsize=16, ha="center", color="black")

        # Save to png
        plt.savefig("{}" "vis_action_{:03d}.png".format(image_path, i))
        plt.clf()

print(Timestamp(), "#################### Visual Pose Complete ####################")
