from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import ast

from base import BaseDataLoader
from .data_preprocess import cars_preprocess


class CARSDataset(Dataset):
    def __init__(self, data, max_seq_len, type="train"):
        self.data = data
        self.max_seq_len = max_seq_len
        self.type = type
        
        if self.type == "train":
            self.X = self.data.drop('rating', axis=1)
            self.y = self.data.rating
        else:
            self.X = self.data
    
    def __getitem__(self, index):
        row = self.X.loc[index]
        
        user, item, year = row[0], row[1], row[5]
        director, writer, genre = self.pad_sequence(ast.literal_eval(row[2])), self.pad_sequence(ast.literal_eval(row[3])), self.pad_sequence(ast.literal_eval(row[4]))
        
        if self.type == "train":
            return torch.IntTensor([user, item] + director + writer + genre + [year]), self.y.loc[index]
        else:
            return torch.IntTensor([user, item] + director + writer + genre + [year])
    
    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, att_list):
        seq = [0] * self.max_seq_len
        if len(att_list) > self.max_seq_len:
            seq = att_list[:self.max_seq_len]
        else:
            seq[:len(att_list)] = att_list
        
        return seq
    
    
class CARSDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, max_seq_len, shuffle=False, num_workers=1):
        
        self.train, self.valid = self.load_context_data(data_dir)
        self.train_dataset = CARSDataset(self.train, max_seq_len, "train")
        self.test_dataset = CARSDataset(self.valid, max_seq_len, "train")
        
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        
        super().__init__(self.train_dataset, **self.init_kwargs)
        
        
    def load_context_data(self, data_dir):
        
        if not os.path.exists(os.path.join(data_dir, "context_train_data.csv")):
            cars_preprocess(data_dir)
        
        return pd.read_csv(os.path.join(data_dir, "context_train_data.csv")), pd.read_csv(os.path.join(data_dir, "context_valid_data.csv"))
    
    def split_validation(self):
        return DataLoader(self.test_dataset, **self.init_kwargs)
