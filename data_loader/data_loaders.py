from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_loader.preprocess import BERT4Rec_preprocess


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


class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len, mask_prob):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        # iterator를 구동할 때 사용됩니다.
        seq = self.user_train[user]
        tokens = []
        labels = []
        for s in seq:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # BERT 학습
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s)  # 학습에 사용
            else:
                tokens.append(s)
                labels.append(0)  # 학습에 사용 X, trivial
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)
    
class BERT4RecDataLoader(DataLoader):
    def __init__(self, **args):
        self.args = args
        self.user_train, self.user_valid = BERT4Rec_preprocess(args)
        
        self.seq_dataset = SeqDataset(self.user_train, args['num_user'], args['num_item'], args['max_len'], args['mask_prob'])
        
        super().__init__(self.seq_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)
        
    def split_validation(self):
        return self.user_valid
    
    def return_data(self):
        return self.user_train, self.user_valid