from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset, DataLoader

from data_loader.preprocess import train_valid_split, make_inter_mat


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class AutoRecDataset(Dataset):
    def __init__(self, args, inter_mat, answers_mat):
        self.args = args
        self.inter_mat = inter_mat
        self.answers = answers_mat.argsort(axis = 1)

    def __len__(self):
        return len(self.inter_mat)

    def __getitem__(self, index):
        user_id = index
        inter_mat = self.inter_mat[user_id]
        answers = self.answers[user_id][-10:]
       
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(inter_mat, dtype=torch.float),
            torch.tensor(answers, dtype=torch.long),
        )

        return cur_tensors
    

class AutoRecDataLoader(DataLoader):
    def __init__(self, **args):
        self.train_set, self.valid_set, self.item_set = train_valid_split(args)
        self.train_mat, self.valid_mat, self.item_mat = make_inter_mat(args['data_dir'], self.train_set, self.valid_set, self.item_set)

        self.train_dataset = AutoRecDataset(args, self.item_mat, self.valid_mat)
        self.args = args

      
        super().__init__(self.train_dataset, batch_size = args['batch_size'], shuffle = True, pin_memory = True)

    def split_validation(self):
        self.eval_dataset = AutoRecDataset(self.args, self.train_mat, self.valid_mat)

        return DataLoader(self.eval_dataset, self.batch_size, shuffle = False, pin_memory = True)
        
    def submission(self):
        self.submission_dataset = AutoRecDataset(self.args, self.item_mat, self.valid_mat)
        return DataLoader(self.submission_dataset, self.batch_size, shuffle = False, pin_memory = True)
    