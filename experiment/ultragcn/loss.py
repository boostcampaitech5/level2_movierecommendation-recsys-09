import torch.nn.functional as F
import torch


def get_betas(model, users, items):
    user_degree = model.constraint_mat['user_degree'].to('cuda')
    item_degree = model.constraint_mat['item_degree'].to('cuda')
    
    weights = 1 + model.lambda_ * (1/user_degree[users]) * torch.sqrt((user_degree[users]+1)/(item_degree[items]+1))
    
    return weights

   
def cal_loss_L(beta_weight, output, target):
    
    loss = F.binary_cross_entropy(output, target.float(), weight=beta_weight, reduction='none')
    
    return loss.sum()


def cal_loss_I(model, users, items):
    omega_mat = model.omega_mat.to('cuda')
    omega_idx = model.omega_idx_mat.to('cuda')
    
    user_embeds = model.user_embeds
    item_embeds = model.item_embeds
    
    item_idx_mat = omega_idx[items].squeeze(1)
    
    e_j = item_embeds(item_idx_mat.int())
    e_u = user_embeds(users).expand(-1, e_j.shape[1], -1)
    
    mm = torch.log((e_j * e_u).sum(-1).sigmoid())
    weight = omega_mat[items].squeeze(1)
    
    loss = (mm * weight).sum(-1)
    
    return -1 * loss.sum()


def norm_loss(model):
    loss = 0.0
    for parameter in model.parameters():
        loss += torch.sum(parameter ** 2)
    return loss / 2


def UltraGCN_loss(model, output, data, target):
    
    users = data[:, 0]
    items = data[:, 1]
    
    beta_weight = get_betas(model, users, items)
    
    pos_idx = torch.nonzero(target)

    loss = cal_loss_L(beta_weight, output, target) 
    loss += cal_loss_I(model, users[pos_idx], items[pos_idx]) * model.gamma
    loss += model.delta * norm_loss(model)

    return loss