import pandas as pd
import numpy as np
import math
from operator import itemgetter
from Evaluation_Metrics import *
from loadDataSet import *
import random

def initLFM(train, F):
    p = dict()
    q = dict()
    for u, items in train.items():
        if u not in p:
            p[u] = [random.random() / math.sqrt(F) for x in range(0, F)]
        
        for i in items.keys():
            if i not in q:
                q[i] = [random.random() / math.sqrt(F) for x in range(0, F)]
    return p, q

def randomSelectNegativeSample(items, item_pools):
    ret = dict()
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(items) * 3):
        item = item_pools[random.randint(0, len(item_pools) - 1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret
	
def predict(u, i, p, q):
    return sum(p[u][f] * q[i][f] for f in range(0, len(p[u])))

def learnLFM(train, F, N_loop, alpha, lambda_arg):
    p, q = initLFM(train, F)
    item_pools = []
    for user, items in train.items():
        for item in items.keys():
            item_pools.append(item)

    for step in range(0, N_loop):
        for u, items in train.items():
            samples = randomSelectNegativeSample(items, item_pools)
            for i, rui in samples.items():
                pui = predict(u, i, p, q)
                eui = rui - pui
                for f in range(0, F):
                    p[u][f] += alpha * (q[i][f] * eui - lambda_arg * p[u][f])
                    q[i][f] += alpha * (p[u][f] * eui - lambda_arg * q[i][f])
        alpha *= 0.9
    return p, q

def Recommend(user, train, p, q, N):
    rank = dict()
    rank_sorted = dict()
    ru = train[user]
    for i in q.keys():
        if i in ru:
            continue
        if i not in rank:
            rank[i] = 0
            for f in range(0, len(p[user])):
                rank[i] += p[user][f] * q[i][f]
    rank_sorted = {i: rui for i, rui in sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]}
    return rank_sorted


train_user_ratings, test_user_ratings = loadDataSet()

# Test offline performance on MovieLens data
# F：Number of latent features
# N_loop: Max iterations
# alpha: learning rate
# lambda_arg:  normalization parameters
# N：Number of recommended items for each user

F = 100
N_loop = 300
alpha = 0.02
lambda_arg = 0.01
p, q = learnLFM(train_user_ratings, F, N_loop, alpha, lambda_arg)

precision, recall, coverage, popularity = LFM_Evaluation(Recommend, train_user_ratings, test_user_ratings, p, q, N)
print(precision, recall, coverage, popularity)





