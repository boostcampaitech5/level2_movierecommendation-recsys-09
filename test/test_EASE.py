import argparse
import torch
import pandas as pd
import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from multiprocessing import Pool, cpu_count
from parse_config import ConfigParser


def predict_by_user(user, group, pred, items, top_k):
    watched_item = set(group['label_item'])
    candidates_item = [item for item in items if item not in watched_item]
    # 안 본 영화의 index를 기준으로 추출
    pred = np.take(pred, candidates_item)
    # 큰 순서대로 정렬하고 top_k개의 index 출력
    res = np.argpartition(pred, -top_k)[-top_k:]
    r = pd.DataFrame(
        {
            "user": [user] * len(res),
            "item": np.take(candidates_item, res),
            "score": np.take(pred, res),
        }
    ).sort_values('score', ascending=False)
    return r


def main(config):
    
    df = pd.read_csv(os.path.join(config['test']['data_dir'], "train_ratings.csv"))
    
    model = None
    with open(os.path.join(config['test']['model_dir'], "model.pickle"), "rb") as f:
        model = pickle.load(f)
        
    users = df['user'].unique()
    items = df['item'].unique()
    items = model.item_enc.transform(items)
    train = df.loc[df.user.isin(users)]
    train['label_user'] = model.user_enc.transform(train.user)
    train['label_item'] = model.item_enc.transform(train.item)
    train_groupby = train.groupby('label_user')
    with Pool(cpu_count()) as p:
        user_preds = p.starmap(
            predict_by_user,
            [(user, group, model.pred[user, :], items, config['test']['top_k']) for user, group in train_groupby],
        )
    pred_df = pd.concat(user_preds)
    pred_df['user'] = model.user_enc.inverse_transform(pred_df['user'])
    pred_df['item'] = model.item_enc.inverse_transform(pred_df['item'])
    
    pred_df = pred_df.drop('score',axis = 1)

    if not os.path.exists(config['test']['submission_dir']):
        os.makedirs(config['test']['submission_dir'])
    pred_df.to_csv(config['test']['submission_dir'] + 'my_ease_{}_{}.csv'.format(model._lambda, config['test']['top_k']), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config_EASE.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)