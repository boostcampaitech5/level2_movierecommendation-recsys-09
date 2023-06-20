import argparse
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from data_loader.data_preprocess import cars_test_preprocess, indexing, unindexing
from data_loader.data_loaders import CARSDataset
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device


def main(config):
    
    device, _ = prepare_device(config['n_gpu'])
    
    data_path = os.path.join(config['test']['data_dir'], "context_test_data.csv")
    
    if not os.path.exists(data_path):
        cars_test_preprocess(config['test']['data_dir'])
    
    data = pd.read_csv(data_path)
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    model_path = config['test']['model_dir']
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to(device)
    model.eval()
    
    pred_list = []
    test_dataset = CARSDataset(data, config['data_loader']['args']['max_seq_len'], "test")
    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'],  shuffle=False)

    for data in test_dataloader:
        rating_pred = model(data.to(device))
        rating_pred = rating_pred.cpu().data.numpy().copy()
        ind = np.argpartition(rating_pred.squeeze(1), -10)[-10:]

        pred_list.append(data[ind, 1])
    
    first_row = True
    recommend_df = pd.DataFrame()
    for i, items in enumerate(pred_list):
        user_rec_df = pd.DataFrame({'user':[i]*len(items), 'item':items})
        if first_row == True:
            recommend_df = user_rec_df
            first_row = False
        else:
            recommend_df = pd.concat([recommend_df, user_rec_df], axis = 0, sort=False)
    
    rating_df = pd.read_csv(os.path.join(config['test']['data_dir'], "train_ratings.csv"))
    for column in ['user', 'item']:
        recommend_df = unindexing(rating_df, recommend_df, column)
    
    recommend_df.to_csv(os.path.join(config['test']['submission_dir'], "nfm_submission.csv"), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='cars_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
