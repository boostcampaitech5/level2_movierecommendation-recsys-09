import argparse
import collections
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader.data_loaders_MVAE import MultiVAEDataset, MultiVAEValidDataset
import torch
from torch.utils.data import DataLoader
from parse_config import ConfigParser
from model.model_MVAE import MultiVAE
import torch.optim as optim
from utils.util_MVAE import Recall_at_k_batch, submission_multi_vae
import wandb
from time import time

import data_loader.data_loaders_MVAE as module_data
os.environ['wandb mode'] = 'offline'

# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    global device
    # device = torch.device("cuda" if config['arch']['args']['device'] == "cuda" else "cpu")
    device = torch.device("cpu")

    # setup data_loader instances
    train_dataset = MultiVAEDataset()
    valid_dataset = MultiVAEValidDataset(train_dataset = train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config['data_loader']['args']['train_batch_size'], drop_last=True, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['data_loader']['args']['valid_batch_size'], drop_last=False, pin_memory=True, shuffle=False)

    # 모델 정의
    n_items = train_dataset.n_items
    model = MultiVAE(config, p_dims=[200, 600, n_items], q_dims=None, dropout=config['arch']['args']['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    #####wandb
    wandb.login()
    logger = config.get_logger('train')
    wandb.init(project=config['name'], config=config, entity="ffm")

    if config['model_name'] == "MultiVAE":
        update_count = 0
        best_r10 = -np.inf

        for epoch in range(1, config['trainer']['epochs'] + 1):
            epoch_start_time = time()
            ###### train ######
            model.train()
            train_loss = 0.0
            start_time = time()

            for batch_idx, batch_data in enumerate(train_loader):
                input_data = batch_data.to(device)
                optimizer.zero_grad()
                if config['trainer']['total_anneal_steps'] > 0:
                    anneal = min(config['trainer']['anneal_cap'], 
                                    1. * update_count / config['trainer']['total_anneal_steps'])
                else:
                    anneal = config['trainer']['anneal_cap']

                recon_batch, mu, logvar = model(input_data)
                
                loss = model.loss_function(recon_batch, input_data, mu, logvar, anneal)
                
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                update_count += 1        

                log_interval = 100
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    elapsed = time() - start_time
                    print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                            'loss {:4.2f}'.format(
                                epoch, batch_idx, len(range(0, 6807, config['data_loader']['args']['train_batch_size'])),
                                elapsed * 1000 / log_interval,
                                train_loss / log_interval))

                    start_time = time()
                    train_loss = 0.0

            ###### eval ######
            recall10_list = []
            recall20_list = []
            total_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_data in valid_loader:
                    input_data, label_data = batch_data # label_data = validation set 추론에도 사용되지 않고 오로지 평가의 정답지로 사용된다. 
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)
                    label_data = label_data.cpu().numpy()
                    
                    if config['trainer']['total_anneal_steps'] > 0:
                        anneal = min(config['trainer']['anneal_cap'], 
                                    1. * update_count / config['trainer']['total_anneal_steps'])
                    else:
                        anneal = config['trainer']['anneal_cap']

                    recon_batch, mu, logvar = model(input_data)

                    loss = model.loss_function(recon_batch, input_data, mu, logvar, anneal)

                    total_loss += loss.item()
                    recon_batch = recon_batch.cpu().numpy()
                    recon_batch[input_data.cpu().numpy().nonzero()] = -np.inf

                    recall10 = Recall_at_k_batch(recon_batch, label_data, 10)
                    recall20 = Recall_at_k_batch(recon_batch, label_data, 20)
                    
                    recall10_list.append(recall10)
                    recall20_list.append(recall20)
            
            total_loss /= len(range(0, 6807, 1000))
            r10_list = np.concatenate(recall10_list)
            r20_list = np.concatenate(recall20_list)
                    
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'r10 {:5.3f} | r20 {:5.3f}'.format(
                        epoch, time() - epoch_start_time, total_loss, np.mean(r10_list), np.mean(r20_list)))
            print('-' * 89)
            
            wandb.log({"valid loss" : total_loss,
            "r20" : np.mean(r20_list), 
            "r10" : np.mean(r10_list)})

            if np.mean(r10_list) > best_r10:
                with open(config['test']['save'], 'wb') as f:
                    torch.save(model, f)
                best_r10 = np.mean(r10_list)

    # inference
    with open(config['test']['save'], 'rb') as f:
        model = torch.load(f)

    submission_multi_vae(args, model, device)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/opt/ml/input/level2_movierecommendation-recsys-09/config/config_MVAE.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)


    sweep_configuration = {
        'method': 'random',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'valid loss'
            },
        'parameters': 
        {
            'lr': {'max': 0.0015, 'min': 0.0005},
            'dropout': {'max': 0.5, 'min': 0.1},
            'embed_dim': {'values': [8,16,32]},
            'batch_size': {'values': [256, 512, 1024]},

         }
    }

    main(config)
