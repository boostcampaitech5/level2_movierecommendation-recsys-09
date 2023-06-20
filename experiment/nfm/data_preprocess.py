import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


def cars_preprocess(data_dir):
    
    # rating data 불러오기
    rating_df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    rating_df.drop(['time'], axis=1, inplace=True)
    
    merged_df = rating_df.copy()
    merged_df['rating'] = 1
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
    
    for train_idx, valid_idx in split.split(merged_df, merged_df['user']):
        train_df = merged_df.loc[train_idx]
        valid_df = merged_df.loc[valid_idx]

    # negative sampling 
    train_df = negative_sampling(merged_df, train_df, 2)
    
    multi_att = ['director', 'writer', 'genre']
    cate_att = ['user', 'item']
    
    for column in multi_att:
        # data 불러오기
        df = pd.read_csv(os.path.join(data_dir, column +"s.tsv"), sep='\t')
    
        # indexing 하기
        df = indexing_wo_0(df, column)
    
        # item별 attribute list를 df에 저장
        df = df.groupby('item')[column].apply(list)
        
        train_df = pd.merge(train_df, df, on=['item'], how='left')
        valid_df = pd.merge(valid_df, df, on=['item'], how='left')
        
    train_df = train_df.fillna("[]")
    valid_df = valid_df.fillna("[]")

    year_df = preprocess_year_data(data_dir)
    train_df = pd.merge(train_df, year_df, on=['item'], how='left') 
    valid_df = pd.merge(valid_df, year_df, on=['item'], how='left') 

    for column in cate_att:
        train_df = indexing(rating_df, train_df, column)
        valid_df = indexing(rating_df, valid_df, column)
        
    train_df.to_csv(os.path.join(data_dir, "context_train_data.csv"), index=False)
    valid_df.to_csv(os.path.join(data_dir, "context_valid_data.csv"), index=False)
    

def preprocess_year_data(data_dir):

    def year(x):
        try:
            return float(x[-5:-1])
        except:
            return float(x[-6:-2])

    year_data = pd.read_csv(os.path.join(data_dir, "years.tsv"), sep='\t')
    title_data = pd.read_csv(os.path.join(data_dir, "titles.tsv"), sep='\t')
    
    year_title_df = year_data.merge(title_data, on=['item'], how='outer')
    year_title_df['year'] = year_title_df['title'].apply(year)
    
    return year_title_df[['item', 'year']]
    

def negative_sampling(rating_df, train_df, num_neg):
    total_items = set(rating_df.item)
    user_group_rating_df = list(rating_df.groupby('user')['item'])
    user_group_train_size = train_df.groupby('user').size()
    
    first_row = True
    user_neg_dfs = pd.DataFrame()
    
    for u, u_items in tqdm(user_group_rating_df):
        pos_items, pos_items_len = set(u_items), user_group_train_size[u]
        
        num_negative = pos_items_len * num_neg
        
        if len(total_items - pos_items) > num_negative:
            neg_items = np.random.choice(list(total_items - pos_items), num_negative, replace=False)
        else:
            neg_items = list(total_items - pos_items)
            num_negative = len(neg_items)
        
        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': neg_items, 'rating': [0]*num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)
            
    return pd.concat([train_df, user_neg_dfs], axis=0, sort=False)


def indexing_wo_0(df, column):
    attribute = sorted(list(set(df[column])))
    att2idx = {v:(i+1) for i,v in enumerate(attribute)} # paddig을 위한 0을 남겨둔다.
    
    df[column] = df[column].map(lambda x: att2idx[x])
    
    return df


def indexing(rating_df, df, column):
    attribute = sorted(list(set(rating_df[column])))
    att2idx = {v:i for i,v in enumerate(attribute)}
    
    df[column] = df[column].map(lambda x: att2idx[x])
    
    return df


def unindexing(rating_df, df, column):
    attribute = sorted(list(set(rating_df[column])))
    att2idx = {v:i for i,v in enumerate(attribute)}
    idx2att = {v:i for i,v in att2idx.items()}
    
    df[column] = df[column].map(lambda x: idx2att[x])
    
    return df
    
    
def cars_test_preprocess(data_dir):
    rating_df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    rating_df.drop(['time'], axis=1, inplace=True)
    
    merged_df = pd.read_csv(os.path.join(data_dir, "my_ease_350_100.csv"))

    multi_att = ['director', 'writer', 'genre']
    cate_att = ['user', 'item']
    
    for column in multi_att:
        # data 불러오기
        df = pd.read_csv(os.path.join(data_dir, column +"s.tsv"), sep='\t')
    
        # indexing 하기
        df = indexing_wo_0(df, column)
    
        # item별 attribute list를 df에 저장
        df = df.groupby('item')[column].apply(list)
        
        merged_df = pd.merge(merged_df, df, on=['item'], how="left")
    
    merged_df = merged_df.fillna("[]")

    year_df = preprocess_year_data(data_dir)
    merged_df = pd.merge(merged_df, year_df, on=['item'], how='left')     

    for column in cate_att:
        merged_df = indexing(rating_df, merged_df, column)
        
    merged_df.to_csv(os.path.join(data_dir, "context_test_data.csv"), index=False)
    
    
def cars_total_test_preprocess(data_dir):
    rating_df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    rating_df.drop(['time'], axis=1, inplace=True)
    
    user = pd.DataFrame({"user": rating_df.user.unique()})
    item = pd.DataFrame({"item": rating_df.item.unique()})
    
    merged_df = user.merge(item, how='cross').sort_values(['user', 'item']).reset_index(drop=True)
    
    multi_att = ['director', 'writer', 'genre']
    cate_att = ['user', 'item']
    
    for column in multi_att:
        # data 불러오기
        df = pd.read_csv(os.path.join(data_dir, column +"s.tsv"), sep='\t')
    
        # indexing 하기
        df = indexing_wo_0(df, column)
    
        # item별 attribute list를 df에 저장
        df = df.groupby('item')[column].apply(list)
        
        merged_df = pd.merge(merged_df, df, on=['item'], how="left")
    
    merged_df = merged_df.fillna("[]")

    year_df = preprocess_year_data(data_dir)
    merged_df = pd.merge(merged_df, year_df, on=['item'], how='left')     

    for column in cate_att:
        merged_df = indexing(rating_df, merged_df, column)
        
    merged_df.to_csv(os.path.join(data_dir, "context_test_data_total.csv"), index=False)