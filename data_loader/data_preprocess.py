import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def ultragcn_preprocess(data_dir, num_neg, validation_split):
    rating_df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    rating_df.drop(['time'], axis=1, inplace=True)
    
    merged_df = rating_df.copy()
    columns = merged_df.columns
    merged_df['rating'] = 1
    
    for column in columns:
        merged_df = indexing(rating_df, merged_df, column)
        
    save_constraint_matrix(data_dir, merged_df)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=123)
    
    for train_idx, valid_idx in split.split(merged_df, merged_df['user']):
        train_df = merged_df.loc[train_idx]
        valid_df = merged_df.loc[valid_idx]

    train_df = negative_sampling(merged_df, train_df, num_neg)
    
    train_df.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(data_dir, "valid_data.csv"), index=False)

    
def negative_sampling(rating_df, train_df, num_neg):
    total_items = set(rating_df.item)
    user_group_rating_df = list(rating_df.groupby('user')['item'])
    
    df_list = []
    for u, u_items in tqdm(user_group_rating_df):
        pos_items = set(u_items)
        
        if len(total_items - pos_items) > num_neg:
            neg_items = np.random.choice(list(total_items - pos_items), num_neg, replace=False)
        else:
            neg_items = list(total_items - pos_items)
            num_neg = len(neg_items)
        
        i_user_neg_df = pd.DataFrame({'user': [u]*num_neg, 'item': neg_items, 'rating': [0]*num_neg})
        df_list.append(i_user_neg_df)
            
    return pd.concat([train_df] + df_list, axis=0, sort=False)


def indexing(rating_df, df, column):
    attribute = sorted(list(set(rating_df[column])))
    att2idx = {v:i for i,v in enumerate(attribute)}
    
    df[column] = df[column].map(lambda x: att2idx[x])
    
    return df


def save_constraint_matrix(data_dir, data):
    
    user_groupby = data.groupby('user').agg({'item':'count'}).sort_values('user').item.to_list()
    item_groupby = data.groupby('item').agg({'user':'count'}).sort_values('item').user.to_list()

    constraint_mat = {"user_degree": torch.Tensor(user_groupby),
                      "item_degree": torch.Tensor(item_groupby)}
    
    with open(os.path.join(data_dir, 'constraint_matrix.pickle'), 'wb') as f:
        pickle.dump(constraint_mat, f)
        

def unindexing(rating_df, df, column):
    attribute = sorted(list(set(rating_df[column])))
    att2idx = {v:i for i,v in enumerate(attribute)}
    idx2att = {v:i for i,v in att2idx.items()}
    
    df[column] = df[column].map(lambda x: idx2att[x])
    
    return df


def ultragcn_total_test_preprocess(data_dir):
    rating_df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    rating_df.drop(['time'], axis=1, inplace=True)
    columns = rating_df.columns
    
    user = pd.DataFrame({"user": rating_df.user.unique()})
    item = pd.DataFrame({"item": rating_df.item.unique()})
    
    merged_df = user.merge(item, how='cross').sort_values(['user', 'item']).reset_index(drop=True)
    
    print('df Done')

    for column in columns:
        merged_df = indexing(rating_df, merged_df, column)
        
    merged_df.to_csv(os.path.join(data_dir, "test_data_total.csv"), index=False)