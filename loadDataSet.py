import pandas as pd
import numpy as np
import random

def loadDataSet():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = r_cols, encoding = 'latin-1', engine = 'python')
    ratings.sort_values('user_id', inplace = True)
    
    def splitData(data, M, k, seed):
        random.seed(seed)
        data['rand'] = np.random.randint(0, M, len(data))
        msk = data['rand'] == k
        test = data[msk]
        train = data[~msk]
        return train, test
    
    train, test = splitData(ratings[['user_id','movie_id']], 8, 1, 123)
    
    train_user_ratings = dict()
    test_user_ratings = dict()
    for uid, mids in train.groupby('user_id'):
        r = dict()
        for mid in mids['movie_id']:
            r[mid] = 1
        train_user_ratings[uid] = r

    for uid, mids in test.groupby('user_id'):
        r = dict()
        for mid in mids['movie_id']:
            r[mid] = 1
        test_user_ratings[uid] = r
        
    pop_uid = []
    for uid in train_user_ratings.keys():
        if uid not in test_user_ratings.keys():
            pop_uid.append(uid)
    for uid in pop_uid:
        train_user_ratings.pop(uid, None)
        
    return train_user_ratings, test_user_ratings

