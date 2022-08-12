# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 2_generate_feature.py
# Created Date: 2022-06-12
# Description:
# =======================================================================

import pandas as pd
import pickle
import utils
import time
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import sys

def feat_hist_items(data):
    def get_season(date_time):
        season = (date_time.month - 1) // 3
        season += (date_time.month == 3)&(date_time.day>=20)
        season += (date_time.month == 6)&(date_time.day>=21)
        season += (date_time.month == 9)&(date_time.day>=23)
        season -= 3*int(((date_time.month == 12)&(date_time.day>=21)))
        return season

    def get_day_period(date_time):
        return date_time.hour // 4

    def cat_view_item(items):
        values_dict = {}
        for item_id in items:
            if values_dict.get(item_id) is None:
                values_dict[item_id] = 1
            else:
                values_dict[item_id] += 1
        x = sorted([(k, v) for k, v in values_dict.items()], key=lambda x: x[1], reverse=True)
        default_value = -1
        if x[0][1] <= 1:  # Items visited at most 0 times
            return (items[0], default_value, default_value, 0)
        else:
            # Count the number of items that have been revisited at least once.
            counter = 0
            for item_id, revisits in x:
                if revisits > 1:
                    counter += 1
                else:
                    break  # As we consider that the list is sorted by the number of revisits.
            return (items[0], x[0][0], x[0][1], counter)


    df = data.copy()
    feat = df[ ['session_id', 'hist_item_ids', 'view_dates'] ].drop_duplicates(subset=['session_id'])

    feat['date'] = feat['view_dates'].apply(lambda x: x[0])
    cols = ['cat_day', 'cat_weekday', 'cat_weekend', 'cat_month', 'cat_season']
    feat['cat_day'] = feat['date'].apply(get_day_period)
    feat['cat_weekday'] = feat['date'].apply(lambda x: x.weekday())
    feat['cat_weekend'] = feat['date'].apply(lambda x: int(x.weekday() > 4))
    feat['cat_month'] = feat['date'].apply(lambda x: x.month - 1)
    feat['cat_season'] = feat['date'].apply(get_season)

    cols += ['cat_near_visit', 'cat_most_visit', 'max_view_count', 'max_item_count']
    feat['cat_near_visit'], feat['cat_most_visit'], feat['max_view_count'], feat['max_item_count'] = \
            zip(*feat['hist_item_ids'].apply(cat_view_item))

    cols += ['hist_len', 'hist_len_unique', 'date_gap_max', 'date_gap_min', 'date_gap_mean', 'date_gap_sum']
    feat['hist_len'] = feat['hist_item_ids'].apply( len )
    feat['hist_len_unique'] = feat['hist_item_ids'].apply(lambda x: len(set(x)))
    feat['view_dates'] = feat['view_dates'].apply(
            lambda x: 0 if len(x) == 1 else [pd.Timedelta(x[i] - x[i-1]).seconds for i in range(1, len(x))])
    feat['date_gap_max'] = feat['view_dates'].apply(lambda x: np.max(x))
    feat['date_gap_min'] = feat['view_dates'].apply(lambda x: np.min(x))
    feat['date_gap_mean'] = feat['view_dates'].apply(lambda x: np.mean(x))
    feat['date_gap_sum'] = feat['view_dates'].apply(lambda x: np.sum(x))

    tmp = df[ ['session_id'] ]
    tmp = tmp.merge(feat[['session_id']+cols], on=['session_id'], how='left')
    return tmp[ cols ]

def feat_item_sum_mean_weight(data):
    df = data.copy()
    score_cols = [column for column in df.columns if column.endswith('_score')]
    df = df[ ['item_id']+score_cols ]
    feat = df[['item_id']]
    df = df.groupby('item_id')[score_cols].agg( ['sum', 'mean'] ).reset_index()
    cols = [ f'item_{j}_{i}' for i in score_cols for j in ['sum', 'mean'] ]
    df.columns = ['item_id' ] + cols
    feat = pd.merge( feat, df, on=['item_id'], how='left')
    feat = feat[ cols ]
    return feat

def feat_item_side_sim(data):
    item_features = pd.read_csv('../data/item_features.csv')
    item_features = item_features.pivot_table(index="item_id", 
        columns="feature_category_id", values="feature_value_id", aggfunc='first').reset_index().fillna(-1)
    side_info = dict(zip(item_features['item_id'], item_features.iloc[:,1:].values))
    pair_info = dict()
    def pair_score(side_feat1, side_feat2):
        cat1 = (side_feat1 != -1)
        cat2 = (side_feat2 != -1)
        cover1 = np.sum(cat1[cat2]) / (np.sum(cat1) + 0.000001)
        cover2 = np.sum(cat2[cat1]) / (np.sum(cat2) + 0.000001)
        match = (side_feat1 == side_feat2) & cat1 & cat2
        match1 = np.sum(match) / (np.sum(cat1) + 0.000001)
        match2 = np.sum(match) / (np.sum(cat2) + 0.000001)
        return [cover1, cover2, match1, match2]
    def gen_feature(row):
        hist = row['hist_item_ids']
        item = row['item_id']
        feats = []
        for j in hist:
            tmp = str(j) + '_' + str(item)
            if tmp in pair_info:
                feat = pair_info[tmp]
            else:
                feat = pair_score(side_info[j], side_info[item])
                pair_info[tmp] = feat
            #feat = pair_info[j][item]
            feats.append(feat)
        mean = np.mean(feats, axis=0)
        max = np.max(feats, axis=0)
        min = np.max(feats, axis=0)
        median = np.median(feats, axis=0)
        return pd.Series(np.concatenate([mean, max, mean, median], axis=0))
    df = data.copy()
    feat = df[ ['hist_item_ids','item_id'] ]
    cols = ['side_sim_'+str(i) for i in range(16)]
    feat[cols] = feat.apply(lambda x: gen_feature(x), axis=1)
    return feat[cols]

def feat_item_side(data):
    item_features = pd.read_csv('../data/item_features.csv')
    item_features = item_features.pivot_table(index="item_id", 
        columns="feature_category_id", values="feature_value_id", aggfunc='first').reset_index().fillna(-1)
    item2feature = dict(zip(item_features['item_id'], item_features.iloc[:,1:].values))
    df = data.copy()
    feat = df[ ['session_id', 'hist_item_ids', 'item_id'] ]
    def normalize_vector(x):
        values = np.array(x, dtype=np.float64)
        norm = np.linalg.norm(values)
        values[:] = values[:] / norm
        return values
    def to_vector(items):
        if not isinstance (items, list):
            items = [items]
        ans = []
        for item in items:
            ans.append(item2feature[item])
        ans = np.array(ans)
        counts = []
        for i in range(ans.shape[1]):
            f = set(ans[:, i])
            if -1 in f:
                count = len(set(f)) - 1
            else:
                count = len(set(f))
            counts.append(count)
        return normalize_vector(counts)

    tmp_view = df[ ['session_id', 'hist_item_ids'] ].drop_duplicates(subset=['session_id'])
    tmp_view['view_feature_vector'] = tmp_view['hist_item_ids'].apply(to_vector)
    tmp_item = df[ ['item_id'] ].drop_duplicates(subset=['item_id'])
    tmp_item['item_feature_vector'] = tmp_item['item_id'].apply(to_vector)

    X = np.array([x for x in list(tmp_view['view_feature_vector']) + list(tmp_item['item_feature_vector'])])
    y = KMeans(n_clusters=20, random_state=2022).fit_predict(X)
    tmp_view['cat_view_cluster'] = y[:len(tmp_view)]
    tmp_item['cat_item_cluster'] = y[len(tmp_view):]
    feat = feat.merge(tmp_view[['session_id', 'cat_view_cluster', 'view_feature_vector']], 
            on=['session_id'], how='left')
    feat = feat.merge(tmp_item[['item_id', 'cat_item_cluster', 'item_feature_vector']], 
            on=['item_id'], how='left')
    #feat['cat_equal_cluster'] = feat.apply(lambda x: 
    #        1 if x['cat_view_cluster'] == x['cat_item_cluster'] else 0, axis=1)
    #feat['cluster_cosine'] = feat.apply(lambda x: np.dot(x['view_feature_vector'], x['item_feature_vector'])/
    #                (np.linalg.norm(x['view_feature_vector']) * np.linalg.norm(x['item_feature_vector'])), 
    #                axis=1)
    cols = ['cat_view_cluster', 'cat_item_cluster', 'view_feature_vector', 'item_feature_vector']
    print(feat[cols])
    return feat[cols]

def feat_static_hist(data):
    df = data.copy()
    df['date'] = df['view_dates'].apply(lambda x: pd.to_datetime(x[-1]))
    df['week'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    feat = df[['item_id', 'week', 'day', 'hour']]
    cur_month = min(df['date'].dt.month.values)

    offset = 2
    if cur_month - offset <= 0:
        start = pd.to_datetime('2020-{:02d}-01 00:00:00.000'.format(cur_month - offset + 12))
    else:
        start = pd.to_datetime('2021-{:02d}-01 00:00:00.000'.format(cur_month - offset))
    end = pd.to_datetime('2021-0{}-01 00:00:00.000'.format(cur_month))
    
    cols = []
    for source in ['sessions', 'purchases']:
        data = pd.read_csv('../data/train_{}.csv'.format(source))
        data['date'] = pd.to_datetime(data['date'])
        data['week'] = data['date'].dt.dayofweek
        data['day'] = data['date'].dt.day
        data['hour'] = data['date'].dt.hour
        data = data[(data['date'] >= start) & (data['date'] < end)]
        feat_tmp = data.groupby('item_id').size()
        feat_tmp.name = '{}_item_count'.format(source)
        feat = feat.merge(feat_tmp, how='left', on='item_id')
        cols.append(feat_tmp.name)
        print(f'feat {feat_tmp.name} fuck done')
        for pivot in ['week', 'day', 'hour']:
            feat_tmp = data.groupby(['item_id', pivot]).size()
            feat_tmp.name = '{}_{}_item_count'.format(source, pivot)
            feat = feat.merge(feat_tmp, how='left', on=['item_id', pivot])
            cols.append(feat_tmp.name)
            print(f'feat {feat_tmp.name} fuck done')
    feat = feat.fillna(0)
    return feat[cols]

def feat_item_pair_sim(data):
    df = data.copy()
    df['date'] = df['view_dates'].apply(lambda x: pd.to_datetime(x[-1]))
    feat = df[['hist_item_ids', 'item_id']]
    cur_month = min(df['date'].dt.month.values)

    offset = 2
    if cur_month - offset <= 0:
        start = pd.to_datetime('2020-{:02d}-01 00:00:00.000'.format(cur_month - offset + 12))
    else:
        start = pd.to_datetime('2021-{:02d}-01 00:00:00.000'.format(cur_month - offset))
    end = pd.to_datetime('2021-0{}-01 00:00:00.000'.format(cur_month))
    
    cols = []
    sessions = pd.read_csv('../data/train_sessions.csv')
    purchases = pd.read_csv('../data/train_purchases.csv')
    purchases.columns = ['session_id', 'buy', 'date']
    data = sessions.merge(purchases[['session_id', 'buy']], on='session_id', how='left')
    data.columns = ['session_id', 'view', 'date', 'buy']
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= start) & (data['date'] < end)]
    feat_tmp = data.groupby(['view', 'buy']).size().reset_index()
    feat_tmp.columns = ['view', 'buy', 'size']
    pair_dict = {str(i)+'_'+str(j): k for i, j, k in zip(feat_tmp['view'], feat_tmp['buy'], feat_tmp['size'])}
    def helper(row):
        views = row['hist_item_ids']
        buy = row['item_id']
        res = []
        for view in views:
            key = str(view) + '_' + str(buy)
            if key in pair_dict:
                res.append(pair_dict[key])
        if len(res) == 0:
            res = [0]
        return res
    feat['pair_sim'] = feat.apply(lambda x: helper(x), axis=1)
    feat['pair_sim_mean'] = feat['pair_sim'].apply(lambda x: np.mean(x))
    feat['pair_sim_sum'] = feat['pair_sim'].apply(lambda x: np.sum(x))
    feat['pair_sim_max'] = feat['pair_sim'].apply(lambda x: np.max(x))
    
    return feat[['pair_sim_mean', 'pair_sim_sum', 'pair_sim_max']]
    
mode = sys.argv[1]
data = pickle.load( open('data/{}.pkl'.format(mode), 'rb') )
print(data.iloc[:10, :6])
good_funcs = [  
                feat_hist_items,
                feat_item_sum_mean_weight,
                feat_item_side_sim,
                feat_static_hist,
                feat_item_pair_sim,
                feat_item_side,
             ]
for func in good_funcs:
    t1 = time.time()
    feat = func(data)
    if type(feat) == pd.DataFrame:
        for col in feat.columns:
            feat[col] = utils.downcast(feat[col])
    feat_path = (func.__name__+'_{}.pkl').format(mode)
    t2 = time.time()
    print('do feature {} use time {} s'.format( func.__name__, t2-t1 ))
    utils.dump_pickle(feat, 'data/' + feat_path)
    t3 = time.time()
    print('save feature {} use time {} s'.format( func.__name__, t3-t2 ))
