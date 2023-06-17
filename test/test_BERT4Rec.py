import argparse
import torch
from tqdm import tqdm
import model.model as module_arch
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
from trainer import BERT4RecTrainer
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def main(config):
    logger = config.get_logger('test')

    # get data
    data_loader = config.init_obj('data_loader', module_data)
    user_train, user_valid = data_loader.return_data()
    
    # get model
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    model = model.load_state_dict(config["test"]["model_dir"])
    
    trainer = BERT4RecTrainer(model, config, None, None, None, None, None)
    result = trainer.test(user_train, user_valid)
    
    pd.DataFrame(result, columns=['user', 'item']).to_csv(config["test"]["output_dir"], index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_BERT4Rec.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
