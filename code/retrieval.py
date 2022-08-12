# -*- coding:utf-8 -*-

#========================================================================
# Author: doubleQ
# File Name: retrieval.py
# Created Date: 2022-06-12
# Description:
# =======================================================================

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from tqdm import tqdm
import math
from scipy.sparse import coo_matrix
from collections import defaultdict
from gensim.models import Word2Vec
import os

import dssm
import utils
import srgnn

# * scores of rules are the bigger the better


class PersonalRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each customer."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (session_id, item_id, method, score)
        """

class ItemCF(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

    def get_item_similarity(self, df):
        sim_item = {}
        source_cnt, target_cnt = defaultdict(int), defaultdict(int)
        for index, row in df.iterrows():
            item = row['item_id']
            if item in self.candidate_set:
                date = row['date']
                view_dates = row['view_dates']
                hist_items = row['hist_item_ids']
                target_cnt[item] += 1
                source_cnt[item] += 1
                for loc, (relate_item, view_date) in enumerate(zip(hist_items, view_dates)):
                    source_cnt[relate_item] += 1
                    sim_item.setdefault(relate_item, {})
                    sim_item[relate_item].setdefault(item, 0)
                    sim_item[relate_item][item] += 1 / math.log(1 + len(hist_items))
                
        sim_item_corr = sim_item.copy()
        for i, related_items in sim_item.items():
            for j, cij in related_items.items():
                sim_item_corr[i][j] = cij / (source_cnt[i] * source_cnt[j])
        return sim_item_corr

    def recommend(self, df, sim_item_corr):
        session_ids, preds, scores = [], [], []
        for index, row in df.iterrows():
            rank = {}
            session_id = row['session_id']
            interacted_items = row['hist_item_ids']
            for loc, i in enumerate(interacted_items):
                if i in sim_item_corr:
                    for j, wij in sorted(sim_item_corr[i].items(), 
                            key=lambda d: d[1], reverse=True)[:self.top_k]:
                        rank.setdefault(j, 0)
                        rank[j] += wij 
            sort_rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)
            pred = [x[0] for x in sort_rank]
            score = [x[1] for x in sort_rank]
            pred = pred[:self.top_k]
            score = score[:self.top_k]
            session_ids.append(session_id)
            preds.append(pred)
            scores.append(score)
        candidates = pd.DataFrame({
                'session_id': session_ids,
                'item_id': preds,
                'score': scores
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "itemcf_retrieval_score"
        return candidates

    def retrieve(self):
        sim_item_corr = self.get_item_similarity(self.history_df)
        candidates = self.recommend(self.target_df, sim_item_corr)
        return candidates

class ItemCFTime(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

    def get_item_similarity(self, df):
        sim_item = {}
        source_cnt, target_cnt = defaultdict(int), defaultdict(int)
        for index, row in df.iterrows():
            item = row['item_id']
            if item in self.candidate_set:
                date = row['date']
                view_dates = row['view_dates']
                hist_items = row['hist_item_ids']
                target_cnt[item] += 1
                source_cnt[item] += 1
                max_delta = pd.Timedelta(date - view_dates[-1]).seconds
                for loc, (relate_item, view_date) in enumerate(zip(hist_items, view_dates)):
                    delta = pd.Timedelta(date - view_date).seconds
                    source_cnt[relate_item] += 1
                    sim_item.setdefault(relate_item, {})
                    sim_item[relate_item].setdefault(item, 0)
                    sim_item[relate_item][item] += 1 * (0.5**(loc+1)) * \
                            (2 - delta/(max_delta+0.000001)) / math.log(1 + len(hist_items))
                
        sim_item_corr = sim_item.copy()
        for i, related_items in sim_item.items():
            for j, cij in related_items.items():
                sim_item_corr[i][j] = cij / ((source_cnt[i] * source_cnt[j]) ** 0.2)
        return sim_item_corr

    def recommend(self, df, sim_item_corr):
        session_ids, preds, scores = [], [], []
        for index, row in df.iterrows():
            rank = {}
            session_id = row['session_id']
            interacted_items = row['hist_item_ids']
            for loc, i in enumerate(interacted_items):
                if i in sim_item_corr:
                    for j, wij in sorted(sim_item_corr[i].items(), 
                            key=lambda d: d[1], reverse=True)[:self.top_k]:
                        rank.setdefault(j, 0)
                        rank[j] += wij * (0.5**(loc+1))
            sort_rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)
            pred = [x[0] for x in sort_rank]
            score = [x[1] for x in sort_rank]
            pred = pred[:self.top_k]
            score = score[:self.top_k]
            session_ids.append(session_id)
            preds.append(pred)
            scores.append(score)
        candidates = pd.DataFrame({
                'session_id': session_ids,
                'item_id': preds,
                'score': scores
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "itemcf_time_retrieval_score"
        return candidates

    def retrieve(self):
        sim_item_corr = self.get_item_similarity(self.history_df)
        candidates = self.recommend(self.target_df, sim_item_corr)
        return candidates

class ItemCFFull(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        corpus,
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

        self.w2v = Word2Vec(corpus, vector_size=32, window=3, min_count=0, sg=1, hs=1, workers=32)
        item_features = pd.read_csv('../data/item_features.csv')
        item_features = item_features.pivot_table(index="item_id", 
          columns="feature_category_id", values="feature_value_id", aggfunc='first').reset_index().fillna(-1)
        self.side_info = dict(zip(item_features['item_id'], item_features.iloc[:,1:].values))

    def get_item_similarity(self, df):
        sim_item = {}
        source_cnt, target_cnt = defaultdict(int), defaultdict(int)
        for index, row in df.iterrows():
            item = row['item_id']
            if item in self.candidate_set:
                date = row['date']
                view_dates = row['view_dates']
                hist_items = row['hist_item_ids']
                target_cnt[item] += 1
                source_cnt[item] += 1
                max_delta = pd.Timedelta(date - view_dates[-1]).seconds
                for loc, (relate_item, view_date) in enumerate(zip(hist_items, view_dates)):
                    delta = pd.Timedelta(date - view_date).seconds
                    source_cnt[relate_item] += 1
                    sim_item.setdefault(relate_item, {})
                    sim_item[relate_item].setdefault(item, 0)
                    sim_item[relate_item][item] += 1 * (0.5**(loc+1)) * (2 - delta/(max_delta+0.000001)) / \
                                                math.log(1 + len(hist_items))
                    sim_item.setdefault(item, {})
                    sim_item[item].setdefault(relate_item, 0)
                    sim_item[item][relate_item] += 0.7 * (0.5**(loc+1)) / math.log(1 + len(hist_items))
                    for loc1, other in enumerate(hist_items):
                        if relate_item != other:
                            sim_item[relate_item].setdefault(other, 0)
                            sim_item[relate_item][other] += 0.5 * (0.5**abs(loc-loc1)) / \
                                                math.log(1 + len(hist_items))

        def helper(source, target):
            match = np.sum((target != -1)[(source == target) & (target != -1)]) / \
                    (np.sum(target != -1) + 0.000001)
            return match

        sim_item_corr = sim_item.copy()
        for i, related_items in sim_item.items():
            for j, cij in related_items.items():
                match = helper(self.side_info[i], self.side_info[j])
                cossim = np.dot(self.w2v.wv[str(i)], self.w2v.wv[str(j)]) / \
                        (np.linalg.norm(self.w2v.wv[str(i)]) * np.linalg.norm(self.w2v.wv[str(j)]))
                sim_item_corr[i][j] = 0.5 * cossim + match * cij / ((source_cnt[i] * source_cnt[j]) ** 0.2)
        return sim_item_corr

    def recommend(self, df, sim_item_corr):
        session_ids, preds, scores = [], [], []
        for index, row in df.iterrows():
            rank = {}
            session_id = row['session_id']
            interacted_items = row['hist_item_ids']
            for loc, i in enumerate(interacted_items):
                if i in sim_item_corr:
                    for j, wij in sorted(sim_item_corr[i].items(), 
                            key=lambda d: d[1], reverse=True)[:self.top_k]:
                        rank.setdefault(j, 0)
                        rank[j] += wij * (0.5**(loc+1))
            sort_rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)
            pred = [x[0] for x in sort_rank]
            score = [x[1] for x in sort_rank]
            pred = pred[:self.top_k]
            score = score[:self.top_k]
            session_ids.append(session_id)
            preds.append(pred)
            scores.append(score)
        candidates = pd.DataFrame({
                'session_id': session_ids,
                'item_id': preds,
                'score': scores
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "itemcf_full_retrieval_score"
        return candidates

    def retrieve(self):
        sim_item_corr = self.get_item_similarity(self.history_df)
        candidates = self.recommend(self.target_df, sim_item_corr)
        return candidates

class W2VCF(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, target_df: pd.DataFrame, 
        corpus,
        candidate_set,
        embed_size=32, top_k=20
    ):

        self.target_df = target_df
        self.corpus = corpus
        self.candidate_set = candidate_set
        self.top_k = top_k
        self.embed_size = embed_size

    def get_item_similarity(self):
        self.w2v = Word2Vec(self.corpus, vector_size=self.embed_size, window=3, min_count=0, sg=1, hs=1, workers=32)
        items = [int(x) for x in self.w2v.wv.key_to_index]
        item2id = {v: k for k, v in enumerate(items)}
        item_embed = np.zeros((len(item2id), self.embed_size))
        for v, i in item2id.items():
            item_embed[i] = self.w2v.wv[str(v)]
        l2norm = np.linalg.norm(item_embed, axis=1, keepdims=True)
        item_embed = item_embed / (l2norm+1e-9)
        split_size = 3000
        split_num = int(item_embed.shape[0] / split_size)
        if item_embed.shape[0] % split_size != 0:
            split_num += 1
        topsim = dict()
        all_idx, all_score = [], []
        for i in range(split_num):
            vec = item_embed[i*split_size:(i+1)*split_size]
            sim = vec.dot(np.transpose(item_embed))
            idx = (-sim).argsort(axis=1)
            idx = idx[:, 1: self.top_k+1]
            sim = -sim
            sim.sort(axis=1)
            score = sim[:,1: self.top_k+1]
            score = -score
            all_idx.append(idx)
            all_score.append(score)
        for i, (idx, score) in enumerate(zip(np.concatenate(all_idx), np.concatenate(all_score))):
            topsim[items[i]] = (np.array(items)[idx], score)
        return topsim

    def recommend(self, df, topsim):
        session_ids, preds, scores = [], [], []
        for index, row in df.iterrows():
            session_id = row['session_id']
            interacted_items = row['hist_item_ids']
            rank = {}
            for loc, i in enumerate(interacted_items):
                if i in topsim:
                    for c, (j, wij) in enumerate(zip(topsim[i][0], topsim[i][1])):
                        if c > self.top_k: break
                        if j in self.candidate_set:
                            rank.setdefault(j, 0)
                            rank[j] += wij * (0.5**(loc+1))
            sort_rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)
            pred = [x[0] for x in sort_rank]
            score = [x[1] for x in sort_rank]
            pred = pred[:self.top_k]
            score = score[:self.top_k]
            session_ids.append(session_id)
            preds.append(pred)
            scores.append(score)
        candidates = pd.DataFrame({
                'session_id': session_ids,
                'item_id': preds,
                'score': scores
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "w2vcf_retrieval_score"
        return candidates

    def retrieve(self):
        w2v_path = 'data/w2v.pkl'
        if os.path.exists(w2v_path):
            topsim = utils.load_pickle(w2v_path)
        else:
            topsim = self.get_item_similarity()
            utils.dump_pickle(topsim, w2v_path)
        candidates = self.recommend(self.target_df, topsim)
        return candidates

class W2V(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, target_df: pd.DataFrame, 
        corpus,
        candidate_set,
        embed_size=32, top_k=20
    ):

        self.target_df = target_df
        self.corpus = corpus
        self.candidate_set = candidate_set
        self.top_k = top_k
        self.embed_size = embed_size

    def get_item_similarity(self):
        self.w2v = Word2Vec(self.corpus, vector_size=self.embed_size, window=3, min_count=0, sg=1, hs=1, workers=32)
        items = [int(x) for x in self.w2v.wv.key_to_index]
        item2id = {v: k for k, v in enumerate(items)}
        item_embed = np.zeros((len(item2id), self.embed_size))
        for v, i in item2id.items():
            item_embed[i] = self.w2v.wv[str(v)]
        l2norm = np.linalg.norm(item_embed, axis=1, keepdims=True)
        item_embed = item_embed / (l2norm+1e-9)
        split_size = 3000
        split_num = int(item_embed.shape[0] / split_size)
        if item_embed.shape[0] % split_size != 0:
            split_num += 1
        topsim = dict()
        all_idx, all_score = [], []
        for i in range(split_num):
            vec = item_embed[i*split_size:(i+1)*split_size]
            sim = vec.dot(np.transpose(item_embed))
            idx = (-sim).argsort(axis=1)
            idx = idx[:, 1: self.top_k+1]
            sim = -sim
            sim.sort(axis=1)
            score = sim[:,1: self.top_k+1]
            score = -score
            all_idx.append(idx)
            all_score.append(score)
        for i, (idx, score) in enumerate(zip(np.concatenate(all_idx), np.concatenate(all_score))):
            topsim[items[i]] = (np.array(items)[idx], score)
        return topsim

    def recommend(self, df, topsim):
        session_ids, preds, scores = [], [], []
        for index, row in df.iterrows():
            session_id = row['session_id']
            interacted_items = row['hist_item_ids']
            rank = {}
            for loc, i in enumerate(interacted_items):
                if i in topsim:
                    for c, (j, wij) in enumerate(zip(topsim[i][0], topsim[i][1])):
                        if c > self.top_k: break
                        if j in self.candidate_set:
                            rank.setdefault(j, 0)
                            rank[j] += wij
            sort_rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)
            pred = [x[0] for x in sort_rank]
            score = [x[1] for x in sort_rank]
            pred = pred[:self.top_k]
            score = score[:self.top_k]
            session_ids.append(session_id)
            preds.append(pred)
            scores.append(score)
        candidates = pd.DataFrame({
                'session_id': session_ids,
                'item_id': preds,
                'score': scores
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "w2v_retrieval_score"
        return candidates

    def retrieve(self):
        w2v_path = 'data/w2v.pkl'
        if os.path.exists(w2v_path):
            topsim = utils.load_pickle(w2v_path)
        else:
            topsim = self.get_item_similarity()
            utils.dump_pickle(topsim, w2v_path)
        candidates = self.recommend(self.target_df, topsim)
        return candidates

class Aribnb(PersonalRetrieveRule):

    def __init__(self, 
        history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        embed_size=32, top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = list(candidate_set)
        self.top_k = top_k
        self.embed_size = embed_size

    def get_item_similarity(self):
        corpus = [[str(i) for i in x] + [str(y)] for x, y in zip(self.history_df['hist_item_ids'], self.history_df['item_id'])]
        self.w2v = Word2Vec(corpus, vector_size=self.embed_size, window=3, min_count=0, sg=1, hs=1, workers=32)
        items = [int(x) for x in self.w2v.wv.key_to_index]
        item2id = {v: k for k, v in enumerate(items)}
        item_embed = np.zeros((len(item2id), self.embed_size))
        for v, i in item2id.items():
            item_embed[i] = self.w2v.wv[str(v)]
        return item_embed, item2id

    def recommend(self, df, embed, item2id):
        seq_embedding = []
        label = []
        for interacted_items in self.target_df['hist_item_ids']:
            idx = [item2id[i] for i in interacted_items if i in item2id]
            if len(idx) > 0:
                embedding = embed[idx].mean(axis=0)
            else:
                embedding = np.zeros((self.embed_size))
            seq_embedding.append(embedding)
        self.candidate_set = [i for i in self.candidate_set if i in item2id]
        item_idx = [item2id[i] for i in self.candidate_set]
        item_embedding = embed[item_idx]
        scores = np.dot((np.array(seq_embedding)), np.transpose(item_embedding))
        topk_data_sort, topk_index_sort = topk_matrix(scores, 100)
        preds = np.array(self.candidate_set)[topk_index_sort]
        
        candidates = pd.DataFrame({
                'session_id': self.target_df['session_id'],
                'item_id': [pred for pred in preds],
                'score': [score for score in topk_data_sort]
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "airbnb_retrieval_score"
        return candidates
        
    def retrieve(self):
        embed, item2id = self.get_item_similarity()
        candidates = self.recommend(self.target_df, embed, item2id)
        return candidates

class W2VEmbed(PersonalRetrieveRule):

    def __init__(self, 
        history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        embed_size=32, top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = list(candidate_set)
        self.top_k = top_k
        self.embed_size = embed_size

    def get_item_similarity(self):
        corpus = [[str(i) for i in x] for x in self.history_df['hist_item_ids']] + \
            [[str(i) for i in x] for x in self.target_df['hist_item_ids']]
        self.w2v = Word2Vec(corpus, vector_size=self.embed_size, window=3, min_count=0, sg=1, hs=1, workers=32)
        items = [int(x) for x in self.w2v.wv.key_to_index]
        item2id = {v: k for k, v in enumerate(items)}
        item_embed = np.zeros((len(item2id), self.embed_size))
        for v, i in item2id.items():
            item_embed[i] = self.w2v.wv[str(v)]
        return item_embed, item2id

    def recommend(self, df, embed, item2id):
        seq_embedding = []
        label = []
        for interacted_items in self.target_df['hist_item_ids']:
            embedding = [embed[item2id[i]] * (0.5**(loc+1)) 
                    for loc, i in enumerate(interacted_items) if i in item2id]
            if len(embedding) > 0:
                embedding = np.array(embedding).sum(axis=0)
            else:
                embedding = np.zeros((self.embed_size))
            seq_embedding.append(embedding)
        self.candidate_set = [i for i in self.candidate_set if i in item2id]
        item_idx = [item2id[i] for i in self.candidate_set]
        item_embedding = embed[item_idx]
        scores = np.dot((np.array(seq_embedding)), np.transpose(item_embedding))
        topk_data_sort, topk_index_sort = topk_matrix(scores, 100)
        preds = np.array(self.candidate_set)[topk_index_sort]
        
        candidates = pd.DataFrame({
                'session_id': self.target_df['session_id'],
                'item_id': [pred for pred in preds],
                'score': [score for score in topk_data_sort]
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "w2v_embed_retrieval_score"
        return candidates
        
    def retrieve(self):
        embed, item2id = self.get_item_similarity()
        candidates = self.recommend(self.target_df, embed, item2id)
        return candidates

class DSSM(PersonalRetrieveRule):
    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

    def retrieve(self):
        candidates = dssm.recommend(self.history_df, self.target_df, self.candidate_set, top_k=self.top_k)
        return candidates

class Popular(PersonalRetrieveRule):
    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

    def retrieve(self):
        group = self.history_df[['item_id']]
        group["count"] = 1
        group = group.groupby('item_id')["count"].sum().reset_index()
        group = group[group['item_id'].isin(self.candidate_set)]
        group['count'] = group['count'] / sum(group['count'])
        group = group.sort_values(['count'], ascending=False)
        pred = list(group['item_id'])[:self.top_k]
        score = list(group['count'])[:self.top_k]
        candidates = pd.DataFrame({
                'session_id': self.target_df['session_id'],
                'item_id': [pred for _ in range(len(self.target_df))],
                'score': [score for _ in range(len(self.target_df))]
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "popular_retrieval_score"
        return candidates

class FeatRetrieval(PersonalRetrieveRule):
    def __init__(self,
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.target_df = target_df
        self.candidate_set = list(candidate_set)
        self.top_k = top_k

    def recommend(self):
        item_features = pd.read_csv('../data/item_features.csv')
        item_features = item_features.pivot_table(index="item_id", 
          columns="feature_category_id", values="feature_value_id", aggfunc='first').reset_index().fillna(-1)
        item2feature = dict(zip(item_features['item_id'], item_features.iloc[:,1:].values))

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

        x_embed = np.array([to_vector(x) for x in self.target_df['hist_item_ids']])
        y_embed = np.array([to_vector(x) for x in self.candidate_set])
        scores = np.dot((np.array(x_embed)), np.transpose(y_embed))
        topk_data_sort, topk_index_sort = topk_matrix(scores, self.top_k)
        preds = np.array(self.candidate_set)[topk_index_sort]
        
        candidates = pd.DataFrame({
                'session_id': self.target_df['session_id'],
                'item_id': [pred for pred in preds],
                'score': [score for score in topk_data_sort]
            })
        candidates = candidates.set_index('session_id').apply(pd.Series.explode).reset_index()
        candidates["method"] = "feat_retrieval_score"
        return candidates

    def retrieve(self):
        candidates = self.recommend()
        return candidates

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

class SRGNN(PersonalRetrieveRule):
    def __init__(
        self, history_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        candidate_set,
        top_k=20
    ):

        self.history_df = history_df
        self.target_df = target_df
        self.candidate_set = candidate_set
        self.top_k = top_k

    def retrieve(self):
        candidates = srgnn.recommend(self.history_df, self.target_df, self.candidate_set, top_k=self.top_k)
        return candidates


