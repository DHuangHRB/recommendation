import pandas as pd
import numpy as np
import math
from operator import itemgetter
from Evaluation_Metrics import *
from loadDataSet import *
import importlib

def UserSimilarity(train):
    #build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    #calc co-rated items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            if u not in N:
                N[u] = 0
            N[u] += 1
            if u not in C:
                C[u] = dict()
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1 / math.log(1 + len(users))
    #calc final similarity matrix W
    W = dict()
    for u, related_users in C.items():
        W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def Recommend(user, train, W, K, N):
    rank = dict()
    rank_sorted = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items(), key = itemgetter(1), reverse = True)[0:K]:
        for i, rvi in train[v].items():
                if i in interacted_items:
                    continue
                if i not in rank:
                    rank[i] = 0
                rank[i] += wuv * rvi
    rank_sorted = {i: rvi for i, rvi in sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]}
    return rank_sorted

train_user_ratings, test_user_ratings = loadDataSet()

# Test offline performance on MovieLens data
# N：Number of recommended items for each user
# K: Number of most relevant users 
# W: Similarity Matrix

W = UserSimilarity(train_user_ratings)
N = 10
for K in [10, 20, 40, 80, 160]:
    precision, recall, coverage, popularity = Evaluation(Recommend, train_user_ratings, test_user_ratings, W, K, N)
    print("K:", K)
    print(precision, recall, coverage, popularity)

