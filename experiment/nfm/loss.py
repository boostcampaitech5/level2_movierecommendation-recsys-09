import torch.nn.functional as F


def BCE_loss(output, target):
    return F.binary_cross_entropy(output.squeeze(1), target.float())
