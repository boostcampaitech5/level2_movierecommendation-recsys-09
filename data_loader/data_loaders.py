from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from data_loader.preprocess import train_valid_split, make_inter_mat

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


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

class CatboostDataset():
    def __init__(self, data_path, data_dir, type):
        self.data = pd.read_csv(data_path)
        if type != "test":
            self.data = self.data.drop(['genre'], axis=1)
            self.data_dir = data_dir
            self.type = type
            self.itemtogenre()
            self.multiLabel('genre')

            self.X, self.y = self.data.drop(['rating', 'director', 'writer', 'genre'], axis=1), self.data.rating
        else:
            self.data_dir = data_dir
            self.type = type
            self.test_data()
            self.item_user_label()

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    #     pass

    def itemtogenre(self):
        genre_data = pd.read_csv(os.path.join(self.data_dir, 'genres.tsv'), sep='\t')
        if self.type != 'test':
            genre_data['item'] = LabelEncoder().fit_transform(genre_data.loc[:, 'item'])
        genre_data = genre_data.groupby('item')['genre'].apply(list)
        self.data = pd.merge(self.data, genre_data, on=['item'], how='left')

    def multiLabel(self, column):
        # MultiLabelBinarizer 객체 생성
        mlb = MultiLabelBinarizer()

        # 원핫 인코딩 수행
        encoded_array = mlb.fit_transform(self.data[column])

        # 인코딩 결과를 데이터프레임으로 변환
        df_encoded = pd.DataFrame(encoded_array, columns=mlb.classes_)

        # 원본 데이터프레임과 인코딩된 데이터프레임을 합치기
        self.data = pd.concat([self.data, df_encoded], axis=1)

    def test_data(self):
        year_data = pd.read_csv(os.path.join(self.data_dir, 'years.tsv'), sep='\t')
        self.data = pd.merge(self.data, year_data, on=['item'], how='left')
        self.itemtogenre()
        self.multiLabel('genre')

    def item_user_label(self):
        ratings_df = pd.read_csv(os.path.join(self.data_dir, 'train_ratings.csv'))
        itemtolabel = ratings_df[['item']].copy()
        usertolabel = ratings_df[['user']].copy()
        usertolabel['user_real'] = usertolabel['user']
        itemtolabel['item_real'] = itemtolabel['item']
        usertolabel['user_label'] = LabelEncoder().fit_transform(usertolabel.loc[:, 'user'])
        itemtolabel['item_label'] = LabelEncoder().fit_transform(itemtolabel.loc[:, 'item'])
        usertolabel = usertolabel.drop_duplicates()
        itemtolabel = itemtolabel.drop_duplicates()
        self.data = pd.merge(self.data, usertolabel, on=['user'], how='left')
        self.data = pd.merge(self.data, itemtolabel, on=['item'], how='left')
        self.data['user'] = self.data['user_label']
        self.data['item'] = self.data['item_label']
        self.final_data = self.data[['user', 'item', 'year', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

    def submission_data(self, y_pred):
        self.data['score'] = y_pred
        self.data['user'] = self.data['user_real']
        self.data['item'] = self.data['item_real']
        self.data = self.data.sort_values(['user', 'score'], ascending=[True, False])
        top_10_scores = self.data.groupby('user').apply(lambda x: x.nlargest(10, 'score')).reset_index(drop=True)
        predict = top_10_scores[['user', 'item']]
        predict.to_csv('output/2step_ease_catboost.csv', index=False)


class CatboostDataLoader():
    def __init__(self, data_path, train_data, valid_data):
        self.data_path = data_path
        self.train_data_path = data_path + train_data
        self.valid_data_path = data_path + valid_data
        # self.train, self.valid = self.load_context_data(self)
        self.train_dataset = CatboostDataset(self.train_data_path, self.data_path, "train")
        self.valid_dataset = CatboostDataset(self.valid_data_path, self.data_path, "valid")

    # def load_context_data(self):
        
        # if not os.path.exists(os.path.join(self.train_data_path)):
            # catboost_preprocess(self.data_path)
        
        # return pd.read_csv(os.path.join(self.train_data_path)), pd.read_csv(os.path.join(self.valid_data_path))
    
    def split_validation(self):
        return self.valid_dataset    