import argparse
import collections
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, AutoRecTrainer
from utils import prepare_device, generate_submission_file, EarlyStopping
from tqdm import tqdm
import wandb
import os

#from data_loader.data_loaders import AutoRecDataset


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    #wandb.login()
    logger = config.get_logger('train')


    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    submission_data_loader = data_loader.submission()
    

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
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
    
    #args.train_matrix = train_mat
    if config['name'] == "AutoRec":
        trainer = AutoRecTrainer(
            model, data_loader, valid_data_loader, None, submission_data_loader, config)
        
        checkpoint = config["wandb_name"] + ".pt"
        early_stopping = EarlyStopping(os.path.join(config["output_dir"], checkpoint), patience=config["patience"], verbose=True)
        
        for epoch in tqdm(range(config["trainer"]["epochs"])):
            trainer.train(epoch)

            scores, _ = trainer.valid(epoch)
            
            # 3. wandb log
            """wandb.log({"recall@5" : scores[0],
                    "ndcg@5" : scores[1],
                    "recall@10" : scores[2],
                    "ndcg@10" : scores[3]})"""

            early_stopping(np.array([scores[2]]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        """trainer.args.train_matrix = item_mat
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        preds = trainer.submission(0)
        print(preds)
        generate_submission_file(args.data_file, preds, args.model_name)"""
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
    args.add_argument('-c', '--config', default='config.json', type=str,
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
    main(config)
