import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def item_encoding(df):
        rating_df = df.copy()

        item_ids = rating_df['item'].unique()
        user_ids = rating_df['user'].unique()
        num_item, num_user = len(item_ids), len(user_ids)

        # user, item indexing
        item2idx = pd.Series(data=np.arange(num_item), index=item_ids) # item re-indexing (0~num_item-1)
        
        user2idx = pd.Series(data=np.arange(num_user), index=user_ids) # user re-indexing (0~num_user-1)

        # dataframe indexing
        rating_df = pd.merge(rating_df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
        rating_df = pd.merge(rating_df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
        rating_df.sort_values(['user_idx', 'time'], inplace=True)
        del rating_df['item'], rating_df['user']

        return rating_df

def train_valid_split(args):
    df = pd.read_csv(args['data_dir'] + 'train/train_ratings.csv')
    df = item_encoding(df)

    items = df.groupby("user_idx")["item_idx"].apply(list)
    # {"user_id" : [items]}
    train_set, valid_set, item_set = {} , {}, {}
    print("train_valid set split by user_idx")

    for uid, item in enumerate(tqdm(items)):

        # 유저가 소비한 item의 12.5% 또는 최대 10 으로 valid_set 아이템 구성
        # num_u_valid_set = 10
        num_u_valid_set = min(int(len(item)*0.125), 10)
        u_valid_set = np.random.choice(item, size=num_u_valid_set, replace=False)
        
        train_set[uid] = list(set(item) - set(u_valid_set))
        valid_set[uid] = u_valid_set.tolist()
        item_set[uid] = list(set(item))

    return train_set, valid_set, item_set
        
def make_inter_mat(data_file, *datasets):
    df = pd.read_csv(data_file +  'train/train_ratings.csv')
    df = item_encoding(df)

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    mat_list = []
    dataset_list = datasets

    for dataset in dataset_list: 
        inter_mat = np.zeros((num_users, num_items))
        for uid, items in tqdm(dataset.items()):
            for item in items:
                inter_mat[uid][item] = 1
        mat_list.append(inter_mat)
   
    # 파일 저장 경로
    train_mat_file = data_file + 'train_mat.npy'
    item_mat_file = data_file + 'item_mat.npy'

    # 같은 파일을 한번 더 저장할 필요는 없으므로
    if not os.path.exists(train_mat_file):
        np.save(train_mat_file, mat_list[0])
    if not os.path.exists(item_mat_file): 
        np.save(item_mat_file, mat_list[2])
    
    return mat_list

