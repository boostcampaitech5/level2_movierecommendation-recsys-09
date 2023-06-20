import torch.nn.functional as F
import torch.nn as nn
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def CE_loss(output, target):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion(output, target)

def MultiVAE_loss(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD
