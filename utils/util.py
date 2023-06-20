import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy import sparse

import torch

from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import wandb
import bottleneck as bn

from operator import getitem
from functools import reduce


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def generate_submission_file(data_file, preds, model_name):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()
    item_ids = rating_df['item'].unique()
    
    if model_name in ['SASRec', 'BERT4Rec']:  
        idx2item = pd.Series(data=item_ids, index=np.arange(len(item_ids))+1)  # item idx -> item id
    else:
        idx2item = pd.Series(data=item_ids, index=np.arange(len(item_ids)))
    

    result = []

    for index, items in enumerate(tqdm(preds)):
        for item in items:
            result.append((users[index], idx2item[item]))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        f"output/{model_name}_submission.csv", index=False
    )



def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def Recall_at_k_batch(X_pred, label_data, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1) 
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (label_data > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall



def submission_multi_vae(config, model, device):
    rating_df = pd.read_csv(os.path.join(config['test']['data_dir'], 'train_ratings.csv'), header=0)

    test_unique_uid = pd.unique(rating_df['user'])
    test_unique_sid = pd.unique(rating_df['item'])
    n_items = len(pd.unique(rating_df['item']))

    show2id = dict((sid, i) for (i, sid) in enumerate(test_unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(test_unique_uid))

    test_rating_df = numerize(rating_df, profile2id, show2id)

    n_users = test_rating_df['uid'].max() + 1
    rows, cols = test_rating_df['uid'], test_rating_df['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                            (rows, cols)), dtype='float64',
                            shape=(n_users, n_items))

    test_data_tensor = torch.FloatTensor(data.toarray()).to(device)

    recon_batch, mu, logvar = model(test_data_tensor)

    id2show = dict(zip(show2id.values(),show2id.keys()))
    id2profile = dict(zip(profile2id.values(),profile2id.keys()))

    result = []

    for user in range(len(recon_batch)):
        rating_pred = recon_batch[user]
        rating_pred[test_data_tensor[user].reshape(-1) > 0] = 0

        idx = np.argsort(rating_pred.detach().cpu().numpy())[-10:][::-1]
        for i in idx:
            result.append((id2profile[user], id2show[i]))
# config['test']['submission_dir'] + 
    pd.DataFrame(result, columns=["user", "item"]).to_csv( 
        "output/" + config['model_name']  + ".csv", index=False
    )

def wandb_sweep(model_name, config):
    if model_name == 'AutoRec' or model_name == 'MVAE':
        for k, v in wandb.config.items():
            config['trainer'][k] = v
    
    elif model_name == 'BERT4Rec':
        for k, v in wandb.config.items():
            config['arch']['args'][k] = v
           
    elif model_name == 'UltraGCN':
        for k, v in wandb.config.items():
            _set_by_path(config, k, v)
        
    return config

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def idx2(args):
    df = pd.read_csv(args['data_dir'] + 'train/train_ratings.csv')
    
    item_ids = df['item'].unique() #영화 리스트
    user_ids = df['user'].unique() #사용자 리스트
    
    idx2item = pd.Series(data= item_ids, index=np.arange(len(item_ids))+ 1)
    idx2user = pd.Series(data = user_ids)
    return idx2user, idx2item