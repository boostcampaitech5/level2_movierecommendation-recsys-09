import torch.nn.functional as F
import torch
import torch.nn as nn


def get_betas(model, users, items):
    user_degree = model.constraint_mat['user_degree'].to('cuda')
    item_degree = model.constraint_mat['item_degree'].to('cuda')
    
    weights = 1 + model.lambda_ * (1/user_degree[users]) * torch.sqrt((user_degree[users]+1)/(item_degree[items]+1))
    
    return weights
    
    
def cal_loss_L(beta_weight, output, target):
    
    loss = F.binary_cross_entropy(output, target.float(), weight=beta_weight, reduction='none')
    
    return loss.sum()


def norm_loss(model):
    loss = 0.0
    for parameter in model.parameters():
        loss += torch.sum(parameter ** 2)
    return loss / 2


def UltraGCN_loss(model, output, data, target):
    
    users = data[:, 0]
    items = data[:, 1]
    
    beta_weight = get_betas(model, users, items)
    
    loss = cal_loss_L(beta_weight, output, target) 
    loss += model.delta * norm_loss(model)

    return loss

  
def nll_loss(output, target):
    return F.nll_loss(output, target)

  
def CE_loss(output, target):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion(output, target)

  
def MultiVAE_loss(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD