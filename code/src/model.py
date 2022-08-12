#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        #hy = newgate + inputgate * (hidden - newgate)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node, embed, feature_ids):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size//2)
        self.embedding.weight.data.copy_(torch.from_numpy(embed))
        #self.embedding.weight.requires_grad = False
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        feature_dim = feature_ids.max() + 1
        self.feature_embed = nn.Embedding(num_embeddings=feature_dim, embedding_dim=self.hidden_size//2)
        self.item_feature = nn.Embedding(feature_ids.shape[0], feature_ids.shape[1])
        self.item_feature.weight.data.copy_(torch.from_numpy(feature_ids))
        self.item_feature.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, candidates=None):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        if candidates is None:
            b = self.embed(trans_to_cuda(torch.arange(self.n_node)).unsqueeze(1)).squeeze(1)
        else:
            b = self.embed(trans_to_cuda(candidates).unsqueeze(1)).squeeze(1)
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def embed(self, ids):
        item_embedding = self.embedding(ids)
        feature_ids = self.item_feature(ids).long()
        feature_ids = feature_ids.view(feature_ids.size(0) * feature_ids.size(1), -1)
        feature_embedding = self.feature_embed(feature_ids)
        mask = (feature_ids != 0).float()
        mean_embedding = feature_embedding * mask[:, :, None]
        mean_embedding = torch.sum(mean_embedding, dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-16)
        mean_embedding = mean_embedding.view(
                item_embedding.size(0), item_embedding.size(1), item_embedding.size(2))
        return torch.cat([item_embedding, mean_embedding], dim=-1)

    def forward(self, inputs, A):
        hidden = self.embed(inputs)
        #hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(2)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, candidates=None):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, candidates=candidates)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    #print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        #if j % int(len(slices) / 5 + 1) == 0:
        #    print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    #print('\tLoss:\t%.3f' % total_loss)

    #print('start validing: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(100)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr

def predict(model, test_data, candidate_set, item2id, top_k=100):
    #print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    scores, preds = [], []
    candidate_set = list(candidate_set)
    id2cand = {k: v for k, v in enumerate(candidate_set)}
    candidates = torch.from_numpy(np.array([item2id[i] for i in candidate_set]))
    with torch.no_grad():
        for i in slices:
            targets, logits = forward(model, i, test_data, candidates=candidates)
            datas, indexs = logits.topk(top_k)
            datas = trans_to_cpu(datas).detach().numpy()
            indexs = trans_to_cpu(indexs).detach().numpy()
            indexs = np.vectorize(id2cand.get)(indexs)
            for score, index in zip(datas, indexs):
                scores.append(score.tolist())
                preds.append(index.tolist())
            
    del model
    torch.cuda.empty_cache()
    return preds, scores
