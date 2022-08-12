# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: 4_deep_train.py
# Created Date: 2022-08-12
# Description:
# =======================================================================

from datetime import datetime
from time import time
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import utils
import os
import math
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler


from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


version = datetime.now().strftime("%m%d%H%M%S")
print('Version: ', version)

lr = 0.001
seed = 2022
max_len = 16
mode = 'train'
merge_block_num = 6
batch_size = 1024 * 4
embed_size = 32
epochs = 6
n_fold = 3
mode = sys.argv[1]
os.makedirs('result', exist_ok=True)

class SeqDataset(Dataset):
    def __init__(self, data):
        super(SeqDataset, self).__init__()
        self.data = data

    def collate(self, batch):
        dense = [[item[name] for name in dense_names] for item in batch]
        return {'session_id': torch.tensor([item['session_id'] for item in batch]),
                'hist_item_ids': torch.tensor([item['hist_item_ids'][:max_len] for item in batch]),
                'length': torch.tensor([item['length'] for item in batch]),
                'item_id': torch.tensor([item['item_id'] for item in batch]),
                'label': torch.tensor([item['label'] for item in batch]),
                'dense': torch.tensor(dense),
                }

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)

class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate
        )

class Dice(nn.Module):
    def __init__(self, input_dim, alpha=0., eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + (1 - p) * self.alpha * X
        return output


class MLP_Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 dropout_rates=0.1,
                 batch_norm=False,
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            dense_layers.append(Dice(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        dense_layers.append(nn.Linear(hidden_units[-1], 1, bias=use_bias))
        #dense_layers.append(nn.Sigmoid())
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list

    def forward(self, inputs):
        return self.dnn(inputs)

class DINAttentionLayer(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 attention_units=[32],
                 dropout_rate=0,
                 batch_norm=False):
        super(DINAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_layer = MLP_Layer(input_dim=4*embedding_dim,
                                         hidden_units=attention_units,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm,
                                         use_bias=True)

    def forward(self, query_item, history_sequence, mask):
        # query_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        query_item = query_item.expand(-1, seq_len, -1)
        attention_input = torch.cat([query_item, history_sequence, query_item - history_sequence,
                                     query_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        attention_weight = attention_weight * mask.float()
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1) # mask by all zeros
        return output

class Model(nn.Module):
    def __init__(self, item_dim, feature_dim, feature_ids, embed_size, num_block=3):
        super(Model, self).__init__()
        self.item_embed = nn.Embedding(num_embeddings=item_dim, embedding_dim=embed_size)
        self.item_embed.weight.data.copy_(torch.from_numpy(w2v_embed))
        self.item_embed.weight.requires_grad = False
        self.feature_embed = nn.Embedding(num_embeddings=feature_dim, embedding_dim=embed_size)
        self.item_feature = nn.Embedding(feature_ids.shape[0], feature_ids.shape[1])
        self.item_feature.weight.data.copy_(torch.from_numpy(feature_ids))
        self.item_feature.weight.requires_grad = False
        #self.attns = nn.ModuleList([SelfAttention(embed_size*2) for _ in range(num_block)])
        #self.convs = nn.ModuleList([Conv(max_len) for _ in range(num_block)])
        self.din_layer = DINAttentionLayer(embedding_dim=embed_size*2)
        self.linear = MLP_Layer(input_dim=embed_size*2+len(dense_names), hidden_units=[64, 32])

    def embed(self, ids):
        item_embedding = self.item_embed(ids)
        feature_ids = self.item_feature(ids).long()
        feature_ids = feature_ids.view(feature_ids.size(0) * feature_ids.size(1), -1)
        feature_embedding = self.feature_embed(feature_ids)
        mask = (feature_ids != 0).float()
        mean_embedding = feature_embedding * mask[:, :, None]
        mean_embedding = torch.sum(mean_embedding, dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-16)
        mean_embedding = mean_embedding.view(
                item_embedding.size(0), item_embedding.size(1), item_embedding.size(2))
        return torch.cat([item_embedding, mean_embedding], dim=-1)

    def forward(self, seqs, mask, idx, dense_feature):
        seq_embed = self.embed(seqs.long())
        item_embed = self.embed(idx.long().unsqueeze(1))
        din_output = self.din_layer(item_embed, seq_embed, mask)
        output = torch.cat([din_output, dense_feature.float()], dim=-1)
        score = self.linear(output).view(-1)
        return score

def predict(model, dev_loader, device):
    batch_iterator = tqdm(dev_loader, disable=False)
    with torch.no_grad():
        predicts = []
        for step, dev_batch in enumerate(batch_iterator):
            length = dev_batch['length']
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            predict = model(dev_batch['hist_item_ids'].to(device), 
                            mask.to(device),
                            dev_batch['item_id'].to(device),
                            dev_batch['dense'].to(device),)
            predict = predict.detach().cpu().tolist()
            predicts += predict
    return predicts

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

item_features = pd.read_csv('../data/item_features.csv')
item_features['feature'] = item_features.apply(
        lambda x: str(x['feature_category_id']) + '-' + str(x['feature_value_id']), axis=1)
item2id = {v: k+1 for k, v in enumerate(sorted(list(set(item_features['item_id']))))}
id2item = {item2id[k]: k for k in item2id}
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

datas = []
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('train', block_id)
    datas.append(utils.load_pickle(data_path))
train = pd.concat(datas)
del datas
gc.collect()
datas = []
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('valid', block_id)
    datas.append(utils.load_pickle(data_path))
valid = pd.concat(datas)
del datas
gc.collect()
datas = []
for block_id in range(merge_block_num):
    data_path = 'data/user_data/lgb_model_{}_{}.pkl'.format('test', block_id)
    datas.append(utils.load_pickle(data_path))
test = pd.concat(datas)
del datas
gc.collect()
dense_names = [i for i in train.columns if i.endswith('retrieval_score') and 'mean' not in i and 'sum' not in i]
dense_names += [ 'side_sim_6', 'side_sim_7', 'hist_len', 'hist_len_unique']
print(dense_names)

corpus = [[str(i) for i in items] for items in 
      list(train[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids']) + 
      list(valid[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids']) + 
      list(test[['session_id', 'hist_item_ids']].drop_duplicates(subset=['session_id'])['hist_item_ids']) 
      ]
w2v = Word2Vec(corpus, vector_size=embed_size, window=3, min_count=0, sg=1, hs=1, workers=32)
w2v_embed = np.zeros((len(item2id)+1, embed_size))
for v, i in item2id.items():
    try:
        w2v_embed[i] = w2v.wv[str(v)]
    except:
        continue

deep_data = 'data/deep.pkl'
if not os.path.exists(deep_data):
    sessions = pd.concat([train, valid, test], ignore_index=True)
    sessions['hist_item_ids'] = sessions['hist_item_ids'].apply(lambda x: [item2id[xx] for xx in x])
    sessions['length'] = sessions['hist_item_ids'].apply(lambda x: min(max_len, len(x)))
    sessions['hist_item_ids'] = sessions.apply(
            lambda x: x['hist_item_ids'][:x['length']] + [0] * (max_len-x['length']), axis=1)
    sessions['item_id'] = sessions['item_id'].apply(lambda x: item2id[x])
    sessions['label'] = sessions['label'].astype(int)
    sessions.to_pickle(deep_data) 
else:
    sessions = utils.load_pickle(deep_data)
print(sessions.sample(n=10))

train = sessions.iloc[:len(train)].reset_index()
valid = sessions.iloc[len(train):-len(test)].reset_index()
test = sessions.iloc[-len(test):].reset_index()

print('now train data_shape: ', train.shape)
print('now valid data_shape: ', valid.shape)
print('now test data_shape: ', test.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(len(item2id)+1, len(feature2id)+1, feature_ids, embed_size)
model.to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
m_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
loss_fn.to(device)

if mode == 'offline':
    
    all_sessions = train['session_id'].unique()
    t = train.groupby(['session_id'])['label'].any()
    has_pos_sessions = list( t[t].index )
    train_sessions = list(set(has_pos_sessions))
    train_index = train[ train['session_id'].isin(train_sessions) ].index
    train = train.loc[ train_index ]

    train_set = SeqDataset(train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_set = SeqDataset(valid)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
    ans = valid[ ['session_id', 'item_id', 'label'] ]
    #ans['pred'] = valid['itemcf_full_score']
    #mrr = get_score(ans)
    #print('Recall mrr:', mrr)

    best_mrr = 0.0
    best_epoch = 0
    for epoch in range(10):
        avg_loss, avg_step = 0.0, 0
        batch_iterator = tqdm(train_loader, disable=False)
        for step, train_batch in enumerate(batch_iterator):
            length = train_batch['length']
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            batch_score = model(train_batch['hist_item_ids'].to(device), 
                                mask.to(device),
                                train_batch['item_id'].to(device),
                                train_batch['dense'].to(device),)
            label = train_batch['label'].to(device)
            batch_loss = loss_fn(batch_score, label.float())
            avg_loss += batch_loss.item()
            avg_step += 1
            batch_loss.backward()
            m_optim.step()
            m_optim.zero_grad()
        ans['pred' ] = predict(model, valid_loader, device)
        mrr = get_score(ans)
        print("Epoch {}, loss: {:.4f}, MRR: {:.4f}".format(epoch+1, avg_loss/avg_step, mrr))

elif mode == 'online':
    test_set = SeqDataset(test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    ans = test[ ['session_id', 'item_id'] ]
    
    data = pd.concat([train, valid]).reset_index(drop=True)
    all_sessions = data['session_id'].unique()
    t = data.groupby(['session_id'])['label'].any()
    has_pos_sessions = list( t[t].index )
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2022)                  
    index = kfold.split(X=all_sessions)

    see = data[ ['session_id', 'item_id', 'label'] ]
    see['pred'] = data['itemcf_full_retrieval_score']
    mrr = get_score(see)
    print('Recall mrr:', mrr)
    for i, (train_sessions_index, valid_sessions_index) in enumerate(index):
        print('Fold:', i)
        train_sessions = all_sessions[ train_sessions_index ]
        train_sessions = list(set(train_sessions) & set(has_pos_sessions))
        train_index = data[ data['session_id'].isin(train_sessions) ].index
        train_data = data.loc[ train_index ]

        valid_sessions = all_sessions[ valid_sessions_index ]
        valid_index = data[ data['session_id'].isin(valid_sessions) ].index
        valid_data = data.loc[ valid_index ]

        train_set = SeqDataset(train_data)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_set = SeqDataset(valid_data)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

        best_mrr = 0.0
        best_epoch = 0
        for epoch in range(epochs):
            avg_loss, avg_step = 0.0, 0
            batch_iterator = tqdm(train_loader, disable=False)
            for step, train_batch in enumerate(batch_iterator):
                length = train_batch['length']
                mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
                batch_score = model(train_batch['hist_item_ids'].to(device), 
                                    mask.to(device),
                                    train_batch['item_id'].to(device),
                                    train_batch['dense'].to(device),)
                label = train_batch['label'].to(device)
                batch_loss = loss_fn(batch_score, label.float())
                avg_loss += batch_loss.item()
                avg_step += 1
                batch_loss.backward()
                m_optim.step()
                m_optim.zero_grad()
        result = predict(model, valid_loader, device)
        see.loc[ valid_index , 'pred' ] = result
        if i == 0:
            ans['pred'] = predict(model, test_loader, device)
        else:
            ans['pred'] += predict(model, test_loader, device)
    mrr = get_score(see)
    print('Rank mrr:', mrr)

    ans['item_id'] = ans['item_id'].apply(lambda x: id2item[x])
    scaler = MinMaxScaler()
    ans['pred'] = ans.groupby('session_id')['pred'].apply(lambda x: 
            scaler.fit_transform(np.array(list(x)).reshape(-1, 1)).reshape(-1)).reset_index().explode('pred').reset_index()['pred']
    ans.to_csv('result/dnn_{}_{:.4}_score.csv'.format(version, mrr), index=False)
    ans = ans.sort_values(by=['session_id','pred'], ascending=False).groupby('session_id')['item_id'].apply(lambda x: list(x)[:100]).reset_index()
    ans.columns = ['session_id', 'item_id']
    ans = ans.explode('item_id')
    ans['rank'] = list(range(1, 101)) * len(set(ans['session_id']))
    board_session = set(pd.read_csv('../data/test_leaderboard_sessions.csv')['session_id'])
    final_session = set(pd.read_csv('../data/test_final_sessions.csv')['session_id'])
    test_leaderboard = ans[ans['session_id'].isin(board_session)]
    final_leaderboard = ans[ans['session_id'].isin(final_session)]
    test_leaderboard.to_csv('result/leaderboard_dnn_{}_{:.4}.csv'.format(version, mrr), index=False)
    final_leaderboard.to_csv('result/final_dnn_{}_{:.4}.csv'.format(version, mrr), index=False)
else:
    print('mode error!') 
