import argparse
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader
from data_loader.data_loaders import UltraGCNDataset
from data_loader.data_preprocess import ultragcn_total_test_preprocess, indexing, unindexing
import model.model as module_arch
from parse_config import ConfigParser


def process_chunk(chunk, user_group_df, model):
    data = chunk.reset_index(drop=True)
    user = data.iloc[0, 0]
    items = user_group_df[user][1]
    
    mask = np.ones(len(data), dtype=bool)
    mask[items] = False
    data = data.iloc[mask].reset_index(drop=True)
    
    test_dataset = UltraGCNDataset(data, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'],  shuffle=False)

    for data in test_dataloader:
        rating_pred = model(data.to('cpu'))
        rating_pred = rating_pred.cpu().data.numpy().copy()
        ind = np.argpartition(rating_pred, -20)[-20:]
    
    items = data[ind, 1]
    
    user_rec_df = pd.DataFrame({'user':[user]*len(items), 'item':items})
    
    return user_rec_df


def main(config):
    
    data_path = os.path.join(config['test']['data_dir'], "context_test_data_total.csv")
    
    if not os.path.exists(data_path):
        ultragcn_total_test_preprocess(config['test']['data_dir'])
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    model_path = os.path.join(config['test']['model_dir'], "model_best.pth")
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to('cpu')
    model.eval()
    
    rating_df = pd.read_csv(os.path.join(config['test']['data_dir'], "train_ratings.csv"))
    for column in ['user', 'item']:
        rating_df = indexing(rating_df, rating_df, column)
    user_group_df = list(rating_df.groupby('user')['item'])
    
    with Pool(cpu_count()) as p:
        chunksize=6807
        pred_dfs = p.starmap(process_chunk, [[chunk[['user', 'item']], user_group_df, model] for i, chunk in tqdm(enumerate(pd.read_csv(data_path, chunksize=chunksize)))])

    recommend_df = pd.concat(pred_dfs).sort_values(['user'])
    
    rating_df = pd.read_csv(os.path.join(config['test']['data_dir'], "train_ratings.csv"))
    for column in ['user', 'item']:
        recommend_df = unindexing(rating_df, recommend_df, column)
    
    recommend_df.to_csv(os.path.join(config['test']['submission_dir'], "ultragcn_submission_20.csv"), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_ultragcn.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)