# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 3_generate_data.py
# Created Date: 2022-06-12
# Description:
# =======================================================================

import numpy as np
import pandas as pd
import utils
import sys
import os

mode = sys.argv[1]
feat_names = [
       'feat_hist_items',
       'feat_item_sum_mean_weight',
       'feat_item_side_sim',
       'feat_static_hist',
       'feat_item_pair_sim',
       'feat_item_side'
        ]

base = utils.load_pickle('data/{}.pkl'.format(mode))
for column in base.columns:
    if column.endswith('_score'):
        base[column] = utils.downcast(base[column])
base['label'] = utils.downcast(base['label'])
print(base.iloc[:10, :6])
feat_paths = [ (i+'_{}.pkl').format(mode) for i in feat_names ]
feat_list = [base]
for feat_path in feat_paths:
    feat = utils.load_pickle( 'data/' + feat_path )
    print(feat_path, ':',feat.shape)
    for col in feat.columns:
        feat[col] = utils.downcast(feat[col])
    feat_list.append( feat )
data = pd.concat(feat_list, axis=1).reset_index(drop=True)
print(data.head())
print('[INFO] after merge feature, data shape is ', data.shape)
print('[INFO] features: ', data.columns)
merge_block_num = 6
block_len = len(data)//merge_block_num
os.makedirs('data/user_data', exist_ok=True)
for block_id in range(merge_block_num):
    l = block_id * block_len
    r = (block_id+1) * block_len
    print('merging block: ', block_id)
    output_path = 'data/user_data/lgb_model_{}_{}.pkl'.format(mode ,block_id)
    if block_id == merge_block_num - 1:
        utils.dump_pickle( data.iloc[l:], output_path )
    else:
        utils.dump_pickle( data.iloc[l:r], output_path )
