import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CE_loss(output, target):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion(output, target)