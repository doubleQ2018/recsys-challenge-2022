# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 5_ranker_train.py
# Created Date: 2022-07-12
# Description:
# =======================================================================

from datetime import datetime
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm.sklearn import LGBMRanker
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import utils
import os
import math
import gc
import sys
version = datetime.now().strftime("%m%d%H%M%S")
print('Version: ', version)

seed = 2022
mode = sys.argv[1]
merge_block_num = 6
n_fold = 3
os.makedirs('result', exist_ok=True)

def modeling(train_X, train_Y, test_X, test_Y, train_baskets, test_baskets, cat_features=None):
    feat_cols = list(train_X.columns)
    ranker = LGBMRanker(
            learning_rate=0.01,
            objective="lambdarank",
            metric="ndcg",
            boosting_type="dart",
            importance_type='gain',
            verbose=10,
            num_threads=10,
            n_estimators=100,
            reg_alpha=0.0, 
            reg_lambda=1,
            subsample=0.7, 
            colsample_bytree=0.7, 
            subsample_freq=1,
        )
    print('Start train and validate...')
    ranker = ranker.fit(
        train_X,
        train_Y,
        categorical_feature=cat_features,
        group=train_baskets,
        eval_group=[test_baskets],
        eval_set=[(test_X, test_Y)],
        eval_at=(100, 200),
        callbacks=[early_stopping(stopping_rounds=5, first_metric_only=True)],
    ) 
    for i in ranker.feature_importances_.argsort()[::-1]:
        print(feat_cols[i], ranker.feature_importances_[i]/ranker.feature_importances_.sum())
    return ranker

def predict(test_X, model):
    print('Start Predict ...')
    block_num = 6
    block_len = len(test_X)//block_num
    predicts = []
    for block_id in range(block_num):
        l = block_id * block_len
        r = (block_id+1) * block_len
        if block_id == block_num - 1:
            predict = model.predict( test_X.iloc[l:])
        else:
            predict = model.predict( test_X.iloc[l:r])
        predicts.append(predict)
    predict = np.concatenate( predicts )
    return predict

def get_score(df):
    df = df.sort_values(by=['session_id','pred'], ascending=False).groupby('session_id')['label'].apply(lambda x: list(x)[:100]).reset_index()
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

drop_list = ['session_id', 'item_id', 'hist_item_ids', 'view_dates', 'label', 
             'view_feature_vector', 'item_feature_vector'
            ]
OPT_ROUNDS = 1000

if mode == 'offline':
    datas = []
    for block_id in range(merge_block_num):
        data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('train', block_id)
        datas.append(utils.load_pickle(data_path))
    train = pd.concat(datas)
    #train['view_dates'] = train['view_dates'].apply(lambda x: x[0])
    #train = train[train['view_dates'] >= '2021-03-01 00:00:00.000'].reset_index(drop=True)
    del datas
    gc.collect()
    datas = []
    for block_id in range(merge_block_num):
        data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('valid', block_id)
        datas.append(utils.load_pickle(data_path))
    test = pd.concat(datas)
    del datas
    gc.collect()
    print('now train data_shape: ', train.shape)
    print('now test data_shape: ', test.shape)
    drop_list += [column for column in train.columns if column.startswith('cat_')]
    cat_features = None

    ans = test[ ['session_id', 'item_id', 'label'] ]
    #ans['pred'] = test['itemcf_full_retrieval_score']
    #mrr = get_score(ans)
    #print('Recall mrr:', mrr)

    all_sessions = train['session_id'].unique()
    t = train.groupby(['session_id'])['label'].any()
    has_pos_sessions = list( t[t].index )
    train_sessions = list(set(has_pos_sessions))
    train_index = train[ train['session_id'].isin(train_sessions) ].index
    train = train.loc[ train_index ]
    train_baskets = train.groupby(['session_id'])['item_id'].count().values
    test_baskets = test.groupby(['session_id'])['item_id'].count().values

    train_X = train.drop( drop_list, axis=1 )
    train_Y = train['label']

    test_X = test.drop( drop_list, axis=1 )
    test_Y = test['label']

    model = modeling(train_X, train_Y, test_X, test_Y, train_baskets, test_baskets, cat_features=cat_features)
    result = predict(test_X, model)
    ans['pred' ] = result
    ans.to_csv('kankan.tsv', sep='\t', index=False)
    mrr = get_score(ans)
    print('Rank mrr:', mrr)
elif mode == 'online':
    datas = []
    for block_id in range(merge_block_num):
        data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('train', block_id)
        datas.append(utils.load_pickle(data_path))
    for block_id in range(merge_block_num):
        data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('valid', block_id)
        datas.append(utils.load_pickle(data_path))
    data = pd.concat(datas).reset_index(drop=True)
    #data['view_dates'] = data['view_dates'].apply(lambda x: x[0])
    #data = data[data['view_dates'] >= '2021-04-01 00:00:00.000'].reset_index(drop=True)
    print(data)
    del datas
    gc.collect()
    print('train data_shape: ', data.shape)
    test_datas = []
    for block_id in range(merge_block_num):
        data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('test', block_id)
        test_datas.append(utils.load_pickle(data_path))
    test_data = pd.concat( test_datas )
    print(test_data)
    del test_datas
    gc.collect()
    print('test data_shape: ', test_data.shape)
    #drop_list += [column for column in test_data.columns if column.startswith('cat_')]
    cat_features = [column for column in test_data.columns if column.startswith('cat_')]

    test_X = test_data.drop( drop_list, axis=1 ) 
    ans = test_data[ ['session_id', 'item_id'] ]

    see = data[ ['session_id', 'item_id', 'label'] ]
    see['pred'] = data['itemcf_full_retrieval_score']
    mrr = get_score(see)
    print('Recall mrr:', mrr)
    all_sessions = data['session_id'].unique()
    t = data.groupby(['session_id'])['label'].any()
    has_pos_sessions = list( t[t].index )
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2022)                  
    index = kfold.split(X=all_sessions)
    for i, (train_sessions_index, valid_sessions_index) in enumerate(index):
        print('Fold:', i)
        train_sessions = all_sessions[ train_sessions_index ]
        train_sessions = list(set(train_sessions) & set(has_pos_sessions))
        train_index = data[ data['session_id'].isin(train_sessions) ].index
        train_data = data.loc[ train_index ]

        valid_sessions = all_sessions[ valid_sessions_index ]
        valid_index = data[ data['session_id'].isin(valid_sessions) ].index
        valid_data = data.loc[ valid_index ]
        print('train:', len(train_data), ', valid:', len(valid_data))
        train_baskets = train_data.groupby(['session_id'])['item_id'].count().values
        valid_baskets = valid_data.groupby(['session_id'])['item_id'].count().values

        train_X = train_data.drop( drop_list, axis=1 )
        train_Y = train_data['label']

        valid_X = valid_data.drop( drop_list, axis=1 )
        valid_Y = valid_data['label']

        model = modeling(train_X, train_Y, valid_X, valid_Y, train_baskets, valid_baskets, cat_features=cat_features)
        result = predict(valid_X, model)
        see.loc[ valid_index , 'pred' ] = result
        if i == 0:
            ans['pred'] = predict(test_X, model)
        else:
            ans['pred'] += predict(test_X, model)
    mrr = get_score(see)
    print('Rank mrr:', mrr)

    scaler = MinMaxScaler()
    ans['pred'] = ans.groupby('session_id')['pred'].apply(lambda x: 
            scaler.fit_transform(np.array(list(x)).reshape(-1, 1)).reshape(-1)).reset_index().explode('pred').reset_index()['pred']
    ans.to_csv('result/lgbranker_{}_{:.4}_score.csv'.format(version, mrr), index=False)
    ans = ans.sort_values(by=['session_id','pred'], ascending=False).groupby(
            'session_id')['item_id'].apply(lambda x: list(x)[:100]).reset_index()
    ans = ans.explode('item_id')
    ans['rank'] = list(range(1, 101)) * len(set(ans['session_id']))
    board_session = set(pd.read_csv('../data/test_leaderboard_sessions.csv')['session_id'])
    final_session = set(pd.read_csv('../data/test_final_sessions.csv')['session_id'])
    test_leaderboard = ans[ans['session_id'].isin(board_session)]
    final_leaderboard = ans[ans['session_id'].isin(final_session)]
    test_leaderboard.to_csv('result/leaderboard_lgbranker_{}_{:.4}.csv'.format(version, mrr), index=False)
    final_leaderboard.to_csv('result/final_lgbranker_{}_{:.4}.csv'.format(version, mrr), index=False)
else:
    print('mode error!')
