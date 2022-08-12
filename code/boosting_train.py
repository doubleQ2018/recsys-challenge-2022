# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: boosting_train.py
# Created Date: 2022-07-12
# Description:
# =======================================================================

from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
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

SEED = 2022
merge_block_num = 6
n_fold = 3
os.makedirs('result', exist_ok=True)

def modeling(clf_name, trn_x, trn_y, val_x, val_y, test_x, cat_features=[]):
    if clf_name == 'lgb':
        clf = lgb
    elif clf_name == 'xgb':
        clf = xgb
    else:
        clf = CatBoostClassifier

    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 8,
            'num_leaves': 60,
            'min_child_weight': 5,
            'lambda_l2': 10,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 4,
            "reg_alpha": 10,
            "reg_lambda": 10,
            'learning_rate': 0.005,
            'seed': SEED,
            'n_jobs':8
        }

        model = clf.train(params, train_matrix, 1000, valid_sets=[train_matrix, valid_matrix],
                          categorical_feature=cat_features, verbose_eval=20, early_stopping_rounds=10)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        importances = pd.DataFrame({'features':model.feature_name(),
                                'importances':model.feature_importance()})
        importances.sort_values('importances',ascending=False,inplace=True)
        print(importances.iloc[:20])

    if clf_name == "xgb":
        train_matrix = clf.DMatrix(trn_x , label=trn_y)
        valid_matrix = clf.DMatrix(val_x , label=val_y)
        test_matrix = clf.DMatrix(test_x)

        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'gamma': 1,
                  'min_child_weight': 1.5,
                  'max_depth': 5,
                  'lambda': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.7,
                  'colsample_bylevel': 0.7,
                  'eta': 0.2,
                  'tree_method': 'exact',
                  'seed': SEED,
                  'nthread': 8,
                  }

        watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=500, evals=watchlist, verbose_eval=30, early_stopping_rounds=50)
        val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)

    if clf_name == "cat":
        params = {
                'n_estimators':5000,
                'learning_rate': 0.05,
                'eval_metric':'AUC',
                'loss_function':'Logloss',
                'random_seed':SEED,
                'metric_period':500,
                'od_wait':500,
                'depth': 8,
                }

        model = clf(**params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=cat_features, use_best_model=True)

        val_pred  = model.predict_proba(val_x)[:,1]
        test_pred = model.predict_proba(test_x)[:,1]

    return val_pred, test_pred

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

model_name = sys.argv[1]
drop_list = ['session_id', 'item_id', 'hist_item_ids', 'view_dates', 'label', 
             'view_feature_vector', 'item_feature_vector'
            ]

datas = []
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('train', block_id)
    datas.append(utils.load_pickle(data_path))
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('valid', block_id)
    datas.append(utils.load_pickle(data_path))
data = pd.concat(datas).reset_index(drop=True)
del datas
gc.collect()
print('train data_shape: ', data.shape)
test_datas = []
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('test', block_id)
    test_datas.append(utils.load_pickle(data_path))
test_data = pd.concat( test_datas )
del test_datas
gc.collect()
print('test data_shape: ', test_data.shape)
drop_list += [column for column in test_data.columns if column.startswith('cat_')]

test_X = test_data.drop( drop_list, axis=1 ) 
ans = test_data[ ['session_id', 'item_id'] ]

see = data[ ['session_id', 'item_id', 'label'] ]
#see['pred'] = data['itemcf_full_retrieval_score']
#mrr = get_score(see)
#print('Recall mrr:', mrr)
all_sessions = data['session_id'].unique()
t = data.groupby(['session_id'])['label'].any()
has_pos_sessions = list( t[t].index )
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)
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

    val_pred, test_pred = modeling(model_name, train_X, train_Y, valid_X, valid_Y, test_X)
    see.loc[ valid_index , 'pred' ] = val_pred
    if i == 0:
        ans['pred'] = test_pred
    else:
        ans['pred'] += test_pred
mrr = get_score(see)
print('Rank mrr:', mrr)

scaler = MinMaxScaler()
ans['pred'] = ans.groupby('session_id')['pred'].apply(lambda x: 
        scaler.fit_transform(np.array(list(x)).reshape(-1, 1)).reshape(-1)).reset_index().explode('pred').reset_index()['pred']
ans.to_csv('result/{}_{}_{:.4}_score.csv'.format(model_name, version, mrr), index=False)
ans = ans.sort_values(by=['session_id','pred'], ascending=False).groupby(
        'session_id')['item_id'].apply(lambda x: list(x)[:100]).reset_index()
ans = ans.explode('item_id')
ans['rank'] = list(range(1, 101)) * len(set(ans['session_id']))
ans.to_csv('result/final_{}_{}_{:.4}.csv'.format(model_name, version, mrr), index=False)
