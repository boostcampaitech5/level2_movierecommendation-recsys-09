import argparse
import collections
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader.data_loaders import MultiVAEDataset, MultiVAEValidDataset
import torch
from torch.utils.data import DataLoader
import model.metric as module_metric
from parse_config import ConfigParser
from model.model_MVAE import MultiVAE
import torch.optim as optim
import model.model as module_arch
from utils import prepare_device, wandb_sweep
from utils.util import Recall_at_k_batch, submission_multi_vae
import wandb
from time import time
from trainer import Trainer, AutoRecTrainer, Trainer_ML, MVAE_Trainer

import model.loss as module_loss
import data_loader.data_loaders as module_data
os.environ['wandb mode'] = 'offline'

# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
import functools

def main(config):

    
    #global device
    device = torch.device("cuda" if config['arch']['args']['device'] == "cuda" else "cpu")
    #device = torch.device("cpu")

        # wandb init
    if config['wandb']:
        wandb.login()
        wandb.init(project=config['name'], entity="ffm", name=config['name'])

    # wandb sweep
    if config['wandb_sweep']:
        config = wandb_sweep(config['name'], config) 

    logger = config.get_logger('train')

    # setup data_loader instances
    if config['name'] != 'MVAE':
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)
    else :
        # setup data_loader instances
        train_dataset = MultiVAEDataset()
        valid_dataset = MultiVAEValidDataset(train_dataset = train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=config['data_loader']['args']['train_batch_size'], drop_last=True, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['data_loader']['args']['valid_batch_size'], drop_last=False, pin_memory=True, shuffle=False)
        # 모델 정의
        model = MultiVAE(config, p_dims=config['arch']['args']['p_dims'], q_dims=None, dropout=config['arch']['args']['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['args']['lr'], weight_decay=0)

###########
    # prepare for (multi-device) GPU training
    if config['name'] != 'Catboost' and config['name'] != 'MVAE':
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # prepare trainer and start train
    if config['name'] == "AutoRec_eval":
        trainer = AutoRecTrainer(
            model, data_loader, valid_data_loader, None, None, config)
        trainer.train_and_validate()
     
    else:
        if config['name'] == "Catboost":
            trainer = Trainer_ML(model, config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader)
        elif config['name'] == "MVAE":
            trainer = MVAE_Trainer(model, config=config,
                        data_loader=train_loader,
                        valid_data_loader=valid_loader, optimizer = optimizer)

        else:
            trainer = Trainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=data_loader,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler)

        trainer.train()



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

    if config["wandb_sweep"]:
        sweep_id = wandb.sweep(
            sweep=config['sweep_configuration'],
            project=config['name']
        )
        wandb.agent(sweep_id=sweep_id, function=functools.partial(main, config), count=1, entity='ffm')
    else: 
        main(config)

    main(config)