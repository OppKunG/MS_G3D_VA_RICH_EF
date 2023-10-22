import torch


#################### * Mean per-joint position error *####################
def mpjpe(predicted, target):
    if predicted.shape != target.shape:
        print("predicted.shape", predicted.shape)  # [16, 3, 300, 25, 2]
        print("target.shape", target.shape)

    assert predicted.shape == target.shape
    error = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    return error
