import train_action_pre as Pre_Train_Teacher_Model
import train_action_post as Train_Student_Model
import vis_pose as Visual_Pose_Model

if __name__ == "__main__":
    # Training model with RTX 4090 for 44 hours

    Pre_Train_Teacher_Model  # About 22 hours

    Train_Student_Model  # About 42 hours

    Visual_Pose_Model  # About 1 minute
