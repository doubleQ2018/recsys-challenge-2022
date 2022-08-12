# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 1_generate_candidate.py
# Created Date: 2022-08-12
# Description:
# =======================================================================

import os
import sys
import gc
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import multiprocessing

from retrieval import *

if sys.argv[1] == 'final':
    test_sessions = pd.read_csv('../data/test_final_sessions.csv')
else:
    test_sessions = pd.read_csv('../data/test_leaderboard_sessions.csv'),

train_sessions = pd.read_csv('../data/train_sessions.csv')
train_purchases = pd.read_csv('../data/train_purchases.csv')
train_sessions['date'] = pd.to_datetime(train_sessions['date'])
test_sessions['date'] = pd.to_datetime(test_sessions['date'])
train_purchases['date'] = pd.to_datetime(train_purchases['date'])

item_features = pd.read_csv('../data/item_features.csv')
item_features = item_features.pivot_table(index="item_id", 
    columns="feature_category_id", values="feature_value_id", aggfunc='first').reset_index().fillna(-1)
side_info = dict(zip(item_features['item_id'], item_features.iloc[:,1:].values))

train_sessions = train_sessions.sort_values(['date'], ascending=False).groupby('session_id').agg(
        hist_item_ids=("item_id", list), view_dates=("date", list)).reset_index()
test_sessions = test_sessions.sort_values(['date'], ascending=False).groupby('session_id').agg(
        hist_item_ids=("item_id", list), view_dates=("date", list)).reset_index()
train_sessions = train_sessions.merge(train_purchases[['session_id', 'item_id', 'date']], 
        how='left', on='session_id')
test_sessions['item_id'] = 'NULL'
test_sessions['date'] = test_sessions['view_dates'].apply(lambda x: x[0])
sessions = pd.concat([train_sessions, test_sessions]).reset_index(drop=True)
corpus = [[str(i) for i in items] for items in list(train_sessions['hist_item_ids']) + list(test_sessions['hist_item_ids'])]

def get_score(data):
    df = data.groupby('session_id')['label'].apply(lambda x: list(x)[:100]).reset_index()
    mrr = []
    for idx, row in df.iterrows():
        session_id = row['session_id']
        pred = row['label']
        r = 0
        for i, pred in enumerate(row['label']):
            if pred == 1:
                r = 1 / (i + 1)
                break
        mrr.append(r)
    mrr = np.mean(mrr)
    return mrr

num_candidate = 200

def helper(x):
    return x.retrieve()

def gen(x, y, z, is_train=True):
    history_df = sessions[(sessions['date'] >= pd.to_datetime(x)) & \
            (sessions['date'] < pd.to_datetime(y))].reset_index(drop=True)
    target_df = sessions[(sessions['date'] >= pd.to_datetime(y)) & \
            (sessions['date'] < pd.to_datetime(z))].reset_index(drop=True)
    if is_train:
        candidate_set = set(target_df['item_id'])
    else:
        candidate_set = set(pd.read_csv('../data/candidate_items.csv')['item_id'])
    methods = [
                DSSM(history_df, target_df, candidate_set, top_k=num_candidate),
                SRGNN(history_df, target_df, candidate_set, top_k=num_candidate),
                Aribnb(history_df, target_df, candidate_set, top_k=num_candidate),
                W2VEmbed(history_df, target_df, candidate_set, top_k=num_candidate),
                W2VCF(target_df, corpus, candidate_set, top_k=num_candidate),
                #ItemCF(history_df, target_df, candidate_set, top_k=num_candidate),
                #ItemCFTime(history_df, target_df, candidate_set, top_k=num_candidate),
                ItemCFFull(history_df, target_df, corpus, candidate_set, top_k=num_candidate),
                Popular(history_df, target_df, candidate_set, top_k=30),
                FeatRetrieval(target_df, candidate_set, top_k=20),
            ]
    pool = multiprocessing.Pool()
    candidates = pool.map(helper, methods)
    for items, method in zip(candidates, methods):
        tmp = items.merge(target_df[['session_id', 'item_id']].rename(columns={"item_id": "label"}, 
            errors="ignore"), on='session_id', how='left')
        tmp['label'] = tmp.apply(lambda x: 1 if x['item_id'] == x['label'] else 0, axis=1)
        hit = 1.0 * sum(tmp['label']) / len(set(tmp['session_id']))
        mrr = get_score(tmp)
        print(method)
        print('hit =', hit)
        print('mrr =', mrr)
    candidates = pd.concat(candidates, ignore_index=True)
    candidates = pd.pivot_table(
            candidates,
            values="score",
            index=["session_id", "item_id"],
            columns=["method"],
            aggfunc=np.sum,
        ).reset_index().fillna(0)
    target_df = target_df[['session_id', 'item_id', 'hist_item_ids', 'view_dates']]
    target_df.columns = ['session_id', 'label', 'hist_item_ids', 'view_dates']
    candidates = candidates.merge(target_df, on='session_id', how='left')
    candidates['label'] = candidates.apply(lambda x: 1 if x['item_id'] == x['label'] else 0, axis=1)
    return candidates

os.makedirs('data', exist_ok=True)
train_set = []
for x, y, z in [
        ('2020-11-01 00:00:00.000', '2021-04-01 00:00:00.000', '2021-05-01 00:00:00.000'),
        ]:
    candidates = gen(x, y, z) 
    train_set.append(candidates)
train = pd.concat(train_set).reset_index(drop=True)
print(train.shape)
tmp = train.groupby('session_id')['label'].sum().reset_index()
print('train hit =', len(tmp[tmp['label'] > 0]) / len(tmp))
tmp = train.groupby('session_id')['label'].count().reset_index()
print('train candidate num =', max(tmp['label']))
train.to_pickle('data/train.pkl')

x, y, z = ('2020-12-01 00:00:00.000', '2021-05-01 00:00:00.000', '2021-06-01 00:00:00.000')
valid = gen(x, y, z) 
print(valid.shape)
tmp = valid.groupby('session_id')['label'].sum().reset_index()
print('valid hit =', len(tmp[tmp['label'] > 0]) / len(tmp))
valid.to_pickle('data/valid.pkl')

x, y, z = ('2021-01-01 00:00:00.000', '2021-06-01 00:00:00.000', '2021-07-01 00:00:00.000')
test = gen(x, y, z, is_train=False) 
print(test.shape)
test.to_pickle('data/test.pkl')
