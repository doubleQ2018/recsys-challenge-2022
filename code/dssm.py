# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: dssm.py
# Created Date: 2022-06-12
# Description:
# =======================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from time import time
import itertools
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


train_batch_size = 256
embed_size = 64
max_len = 32
epochs = 50
lr = 1e-3

class SeqDataset(Dataset):
    def __init__(self, data):
        super(SeqDataset, self).__init__()
        self.data = data

    def collate(self, batch):
        return {'session_id': torch.tensor([item['session_id'] for item in batch]),
                'hist_item_ids': torch.tensor([item['hist_item_ids'] for item in batch]),
                'length': torch.tensor([item['length'] for item in batch]),
                'item_id': torch.tensor([item['item_id'] for item in batch]),
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

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.num_attention_heads = 4
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask[:, None, None, :]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer + query

class Conv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        #outputs = outputs.permute(0, 2, 1)
        outputs += inputs
        return outputs
 
class Model(nn.Module):
    def __init__(self, item_dim, feature_dim, feature_ids, embed_size, num_block=3):
        super(Model, self).__init__()
        self.item_embed = nn.Embedding(num_embeddings=item_dim, embedding_dim=embed_size)
        self.feature_embed = nn.Embedding(num_embeddings=feature_dim, embedding_dim=embed_size)
        self.item_feature = nn.Embedding(feature_ids.shape[0], feature_ids.shape[1])
        self.item_feature.weight.data.copy_(torch.from_numpy(feature_ids))
        self.item_feature.weight.requires_grad = False
        self.attns = nn.ModuleList([SelfAttention(embed_size*2) for _ in range(num_block)])
        self.convs = nn.ModuleList([Conv(max_len) for _ in range(num_block)])

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
        #mask = (feature_ids == 0).float() * (-1e10)
        #max_embedding = feature_embedding + mask[:, :, None]
        #max_embedding, _ = torch.max(max_embedding, dim=1)
        #max_embedding = max_embedding.view(
        #        item_embedding.size(0), item_embedding.size(1), item_embedding.size(2))
        return torch.cat([item_embedding, mean_embedding], dim=-1)
        
    def seq_encoder(self, seqs, mask):
        output = self.embed(seqs.long())
        mask = mask.float()
        attention_mask = (1.0 - mask) * -10000.0
        for attn, conv in zip(self.attns, self.convs):
            output = attn(F.normalize(output, p=2, dim=-1), output, attention_mask)
            output = conv(F.normalize(output, p=2, dim=-1))
            output = F.normalize(output, p=2, dim=-1)
            output = output * mask[:, :, None]
        embed = output.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return embed

    def item_encoder(self, idx):
        return self.embed(idx.long().unsqueeze(1)).squeeze(1)

    def forward(self, seqs, mask, idx):
        seq_embed = self.seq_encoder(seqs, mask)
        item_embed = self.item_encoder(idx)
        score = torch.matmul(seq_embed, torch.transpose(item_embed, 0, 1))
        return score

def dev(model, dev_loader, device, item_idx):
    mrr, size = 0, 0
    with torch.no_grad():
        label = []
        seq_embedding = []
        item_embedding = model.item_encoder(
                torch.Tensor(item_idx).type(torch.LongTensor).to(device)).detach().cpu().numpy()
        for step, dev_batch in enumerate(dev_loader):
            length = dev_batch['length']
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            seq_embed = model.seq_encoder(dev_batch['hist_item_ids'].to(device), mask.to(device))
            seq_embedding += seq_embed.detach().cpu().tolist()
            label += dev_batch['item_id'].tolist()
    scores = np.dot((np.array(seq_embedding)), np.transpose(item_embedding))
    topk_data_sort, topk_index_sort = topk_matrix(scores, 100)
    preds = np.array(item_idx)[topk_index_sort]
    for pred, l in zip(preds, label):
        rank = 0
        for i, p in enumerate(pred):
            if p == l:
                rank = i+1
                break
        if rank > 0:
            mrr += 1.0/rank
        if l in item_idx:
            size += 1
    return mrr / size

def topk_matrix(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def predict(model, test_loader, device, id2item, item_idx, top_k=100):
    session_ids, seq_embedding = [], []
    with torch.no_grad():
        item_embedding = model.item_encoder(
                torch.Tensor(item_idx).type(torch.LongTensor).to(device)).detach().cpu().numpy()
        for step, test_batch in enumerate(test_loader):
            session_id = test_batch['session_id'].tolist()
            length = test_batch['length']
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            seq_embed = model.seq_encoder(test_batch['hist_item_ids'].to(device), mask.to(device))
            seq_embedding += seq_embed.detach().cpu().tolist()
            session_ids += session_id
    scores = np.dot((np.array(seq_embedding)), np.transpose(item_embedding))
    topk_data_sort, topk_index_sort = topk_matrix(np.array(scores), top_k)
    preds = np.array(item_idx)[topk_index_sort]
    candidates = pd.DataFrame({
            'session_id': session_ids,
            'item_id': np.vectorize(id2item.get)(preds).tolist(),
            'score': list(topk_data_sort)
            })
    candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
    candidates["method"] = "dssm_score"
    return candidates

def recommend(history_df, target_df, candidate_set, top_k=100): 

    train_sessions = history_df.copy()
    test_sessions = target_df.copy()
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

    train_sessions['item_id'] = train_sessions['item_id'].apply(lambda x: item2id[x])
    test_sessions['item_id'] = 0
    sessions = pd.concat([train_sessions, test_sessions], ignore_index=True)
    sessions['hist_item_ids'] = sessions['hist_item_ids'].apply(lambda x: [item2id[xx] for xx in x])
    sessions['length'] = sessions['hist_item_ids'].apply(lambda x: min(max_len, len(x)))
    sessions['hist_item_ids'] = sessions.apply(
            lambda x: x['hist_item_ids'][:x['length']] + [0] * (max_len-x['length']), axis=1)
    
    index = np.arange(len(train_sessions))
    np.random.shuffle(index)
    split = int(len(index) * 0.1)
    train = sessions.iloc[index[split:]].reset_index(drop=True)
    valid = sessions.iloc[index[:split]].reset_index(drop=True)
    test = sessions.iloc[len(train_sessions):].reset_index(drop=True)
    train_set = SeqDataset(train)
    train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)
    valid_set = SeqDataset(valid)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1024, shuffle=False, num_workers=0)
    test_set = SeqDataset(test)
    test_loader = DataLoader(dataset=test_set, batch_size=1024, shuffle=False, num_workers=0)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Model(len(item2id)+1, len(feature2id)+1, feature_ids, embed_size)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    m_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn.to(device)
    item_idx = [item2id[x] for x in candidate_set]

    best_mrr = 0.0
    best_epoch = 0
    os.makedirs('model', exist_ok=True)
    save_path = 'model/dssm.pth'
    for epoch in range(epochs):
        avg_loss, avg_step = 0.0, 0
        for step, train_batch in enumerate(train_loader):
            length = train_batch['length']
            mask = torch.arange(max_len).expand(length.size(0), max_len) < length.unsqueeze(1)
            batch_score = model(train_batch['hist_item_ids'].to(device), 
                                mask.to(device),
                                train_batch['item_id'].to(device))
            label = torch.arange(train_batch['item_id'].size(0)).type(torch.LongTensor).to(device)
            batch_loss = loss_fn(batch_score, label)
            #if torch.cuda.device_count() > 1:
            #    batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            avg_step += 1
            batch_loss.backward()
            m_optim.step()
            m_optim.zero_grad()
        mrr = dev(model, valid_loader, device, item_idx)
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
            #if torch.cuda.device_count() > 1:
            #    torch.save(model.module.state_dict(), save_path)
            #else:
            torch.save(model.state_dict(), save_path)
        if epoch - best_epoch >= 3:
            break
    state_dict = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return predict(model, test_loader, device, id2item, item_idx, top_k=top_k)

