import math

def Evaluation(Recommend, train, test, W, K, N):
    hit = 0
    all_tu = 0
    all_ru = 0
    recommmend_items = set()
    all_items = set()
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret_pop = 0
    n_pop = 0
    
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        
        tu = test[user].keys()
        rank = Recommend(user, train, W, K, N)
        for item, pui in rank.items():
            recommmend_items.add(item)
            ret_pop += math.log(1 + item_popularity[item])
            n_pop += 1
            
            if item in tu:
                hit += 1
        
        all_tu += len(tu)
        all_ru += len(rank)
    
    ret_pop /= n_pop
        
    return [hit / all_ru, hit / all_tu, len(recommmend_items) / len(all_items), ret_pop]

def LFM_Evaluation(Recommend, train, test, p, q, N):
    hit = 0
    all_tu = 0
    all_ru = 0
    recommmend_items = set()
    all_items = set()
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret_pop = 0
    n_pop = 0
    
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        
        tu = test[user].keys()
        rank = Recommend(user, train, p, q, N)
        for item, pui in rank.items():
            recommmend_items.add(item)
            ret_pop += math.log(1 + item_popularity[item])
            n_pop += 1
            
            if item in tu:
                hit += 1
        
        all_tu += len(tu)
        all_ru += len(rank)
    
    ret_pop /= n_pop
        
    return [hit / all_ru, hit / all_tu, len(recommmend_items) / len(all_items), ret_pop]

def Recall(Recommend, train, test, W, K, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user].keys()
        n_recommend = len(test[user])
        rank = Recommend(user, train, W, K, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / all

def Precision(Recommend, train, test, W, K, N):
    hit = 0 
    all = 0
    for user in train.keys():
        tu = test[user].keys()
        rank = Recommend(user, train, W, K, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / all

def Coverage(Recommend, train, test, W, K, N):
    recommmend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = Recommend(user, train, W, K, N)
        for item, pui in rank.items():
            recommmend_items.add(item)
    return len(recommmend_items) / len(all_items)

def Popularity(Recommend, train, test, W, K, N):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret_pop = 0
    n_pop = 0
    for user in train.keys():
        rank = Recommend(user, train, W, K, N)
        for item, pui in rank.items():
            ret += math.log(1 + item_popularity[item])
            n_pop += 1
    ret_pop /= n_pop
    return ret_pop
