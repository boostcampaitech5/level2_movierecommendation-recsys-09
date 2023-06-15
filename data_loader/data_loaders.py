from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from .data_preprocess import ultragcn_preprocess


class UltraGCNDataset(Dataset):
    def __init__(self, data, type="train"):
        self.data = data
        self.type = type
        
        if self.type == "test":
            self.X = self.data
        else:
            self.X = self.data.drop('rating', axis=1)
            self.y = self.data.rating
        
    def __getitem__(self, index):
        
        if self.type == "test":
            return self.X.loc[index].values
        else:
            return self.X.loc[index].values, self.y.loc[index]
    
    def __len__(self):
        return len(self.data)      


class UltraGCNDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, num_neg, shuffle=False, num_workers=1, validation_split=0.0):
        
        self.train, self.valid = self.load_context_data(data_dir, num_neg, validation_split)
        self.train_dataset = UltraGCNDataset(self.train, "train")
        self.valid_dataset = UltraGCNDataset(self.valid, "valid")
        
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        
        super().__init__(self.train_dataset, **self.init_kwargs)
        
        
    def load_context_data(self, data_dir, num_neg, validation_split):
        
        if not os.path.exists(os.path.join(data_dir, "train_data.csv")):
            ultragcn_preprocess(data_dir, num_neg, validation_split)
        
        return pd.read_csv(os.path.join(data_dir, "train_data.csv")), pd.read_csv(os.path.join(data_dir, "valid_data.csv"))
    
    def split_validation(self):
        return DataLoader(self.valid_dataset, **self.init_kwargs)