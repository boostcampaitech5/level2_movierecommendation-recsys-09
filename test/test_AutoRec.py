import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from trainer import AutoRecTrainer

from utils import generate_submission_file

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances 
    data_loader = config.init_obj('data_loader', module_data).submission()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    
    state_dict = torch.load(os.path.join(config['output_dir'], config['wandb_name'] + ".pt"))
    
    
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    trainer = AutoRecTrainer(
            model, None, None, None, data_loader, config)
    trainer.model.load_state_dict(state_dict)
    preds = trainer.submission(0)
    print(preds)
    data_file = config['data_loader']['args']['data_dir'] + '/train/train_ratings.csv'
    generate_submission_file(data_file, preds, config["name"])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_AutoRec.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
