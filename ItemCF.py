import pandas as pd
import numpy as np
import math
from operator import itemgetter
from Evaluation_Metrics import *
from loadDataSet import *
import importlib

def ItemSimilarity(train):
    #calc co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            if i not in N:
                N[i] = 0
            N[i] += 1
            if i not in C:
                C[i] = dict()
            for j in items:
                if i == j:
                    continue
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1 
    #calc final similarity matrix W
	W = dict()
    for i, related_items in C.items():
        W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W    

def Recommend(user, train, W, K, N):
    rank = dict()
    rank_sorted = dict()
    ru = train[user]
    for i, rui in ru.items():
        for j, wji in sorted(W[i].items(), key = itemgetter(1), reverse = True)[0:K]:
            if j in ru:
                continue
            if j not in rank:
                rank[j] = 0
            rank[j] += wji * rui
    rank_sorted = {i: rui for i, rui in sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]}
    return rank_sorted

train_user_ratings, test_user_ratings = loadDataSet()

# Test offline performance on MovieLens data
# N：Number of recommended items for each user
# K: Number of most relevant items 
# W: Similarity Matrix

W = ItemSimilarity(train_user_ratings)
N = 10
for K in [10, 20, 40, 80, 160]:
    precision, recall, coverage, popularity = Evaluation(Recommend, train_user_ratings, test_user_ratings, W, K, N)
    print("K:", K)
    print(precision, recall, coverage, popularity)

