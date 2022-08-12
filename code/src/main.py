#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
import pandas as pd
from gensim.models import Word2Vec
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=600, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
opt = parser.parse_args()
print(opt)


def main():
    item_features = pd.read_csv('../../data/item_features.csv')
    item2id = {v: k for k, v in enumerate(sorted(list(set(item_features['item_id']))))}
    id2item = {item2id[k]: k for k in item2id}
    item_features['feature'] = item_features.apply(
            lambda x: str(x['feature_category_id']) + '-' + str(x['feature_value_id']), axis=1)
    feature2id = {v: k+1 for k, v in enumerate(sorted(list(set(item_features['feature']))))}
    item_features['item_id'] = item_features['item_id'].apply(lambda x: item2id[x])
    item_features['feature'] = item_features['feature'].apply(lambda x: feature2id[x])
    item_features = item_features.sort_values(['feature_category_id'], ascending=True)\
            .groupby('item_id')['feature'].apply(list).reset_index()
    feature_len = max(item_features['feature'].apply(lambda x: len(x)))
    item_features['feature'] = item_features['feature'].apply(lambda x: x + [0] * (feature_len-len(x)))
    item_features = item_features.sort_values(['item_id'], ascending=True)
    feature_ids = np.array([x for x in item_features['feature'].values])
    feature_ids = np.concatenate((np.zeros((1, feature_len), dtype=int), feature_ids), axis=0)

    train_sessions = pd.read_csv('../../data/train_sessions.csv')
    test_sessions = pd.read_csv('../../data/test_leaderboard_sessions.csv')
    train_purchases = pd.read_csv('../../data/train_purchases.csv')
    candidate_items = list(pd.read_csv('../../data/candidate_items.csv')['item_id'])
    candidate_ids = np.array([item2id[x] for x in candidate_items])

    train_sessions = train_sessions.sort_values(['date'], ascending=True).groupby('session_id').agg(
        hist_item_ids=("item_id", list), date=("date", lambda x: x.tolist()[-1].split()[0])).reset_index()
    test_sessions = test_sessions.sort_values(['date'], ascending=True).groupby('session_id').agg(
        hist_item_ids=("item_id", list), date=("date", lambda x: x.tolist()[-1].split()[0])).reset_index()

    train_sessions = train_sessions.merge(train_purchases[['session_id', 'item_id']], 
            how='left', on='session_id')
    train_sessions['item_id'] = train_sessions['item_id'].apply(lambda x: item2id[x])
    test_sessions['item_id'] = 0
    sessions = pd.concat([train_sessions, test_sessions], ignore_index=True)
    sessions['hist_item_ids'] = sessions['hist_item_ids'].apply(lambda x: [item2id[xx] for xx in x])
    
    train = sessions[(sessions['date'] >= '2021-01-01') & 
                           (sessions['date'] < '2021-05-01')].reset_index(drop=True)
    valid = sessions[(sessions['date'] >= '2021-05-01') &
                           (sessions['date'] < '2021-06-01')].reset_index(drop=True)
    test = sessions[sessions['date'] >= '2021-06-01'].reset_index(drop=True)

    n_node = len(item2id)
    corpus = [[str(i) for i in items] for items in
      list(train[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids']) +
      list(valid[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids']) +
      list(test[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids'])
      ]
    w2v = Word2Vec(corpus, vector_size=opt.hiddenSize, window=3, min_count=0, sg=1, hs=1, workers=32)
    w2v_embed = np.zeros((n_node, opt.hiddenSize//2))
    for v, i in item2id.items():
        try:
            w2v_embed[i] = w2v.wv[str(v)]
        except:
            continue

    train_data = (train['hist_item_ids'].to_list(), train['item_id'].to_list())
    test_data = (valid['hist_item_ids'].to_list(), valid['item_id'].to_list())
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    model = trans_to_cuda(SessionGraph(opt, n_node, w2v_embed, feature_ids))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@100:\t%.4f\tMMR@100:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
