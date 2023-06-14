import json
import math
import os
import random

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from tqdm import tqdm
import scipy

import torch
from scipy.sparse import csr_matrix

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


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


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items, model_name):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    if model_name in ['TFIDF', 'COSINE', 'BM25']:
        rating_matrix = csr_matrix((data.astype(np.float32), (row, col)), shape=(num_users, num_items))
    else:
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


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

def generate_implicit_df(pos_dataset, dataset):
    # df = pd.read_csv(data_file)
    # df = item_encoding(df, model_name)
    
    pos_dataset = pos_dataset
    dataset = dataset

    user_ids = dataset.keys()
    # item_ids = df['item_idx'].unique()

    implicit_df = dict()
    implicit_df['user_idx'] = list()
    implicit_df['item_idx'] = list()
    implicit_df['label'] = list()
    # user_dict = dict()
    # item_dict = dict()


    for user_id in tqdm(user_ids):
        # user_dict[u] = user_id
        item_ids = dataset[user_id]
        if pos_dataset:
            pos_item_ids = pos_dataset[user_id]
        for item_id in item_ids:
            # if i not in item_dict:
                # item_dict[i] = item_id
            implicit_df['user_idx'].append(user_id)
            implicit_df['item_idx'].append(item_id)
            if pos_dataset:
                if item_id in pos_item_ids:
                    implicit_df['label'].append(1)
                else:
                    implicit_df['label'].append(0)
            else:
                implicit_df['label'].append(0)

    implicit_df = pd.DataFrame(implicit_df)

    return implicit_df

