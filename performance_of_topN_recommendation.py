#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def get_hit_arrays(
            train_user_ids, train_item_ids, train_values
           ,test_user_ids, test_item_ids, test_values
           ,model
           ,good_score_threshold=5
           ,n_random_selected_item_ids=1000
           ,remove_still_interaction_from_test=False
           ,topN=[5,10], random_seed=None, **fit_args):
    """
    This is a implementation of (*1) which define a metric for 
    Top-N Recommendation Tasks.    
    
    (*1) Paolo 2010: Performance of Recommender Algorithms on Top-N Recommendation Tasks
    
    INPUT:
    -------------------------------
        train_user_ids, train_item_ids, train_values [array like]: 
            dataset for train.
        test_user_ids, test_item_ids, test_values [array like]: 
            dataset for test.
        model : 
            CF model which has self.fit(user_ids, item_ids, values) and self.predict(user_ids, item_ids).
        good_score_threshold [numeric]:
            to aggregate recall and precision for Collaborative Filtering.
            (*1) evaluates #hit on [test_user_id, test_item_id] having test_value > good_score_threshold.
        n_random_selected_item_ids [int]:
            to aggregate recall and precision for Collaborative Filtering.
        remove_still_interaction_from_test [True/False]:
            This should set False, if you want to run along with (*1).
            But, I think this should be set True...
        topN [list of int]:
            to aggregate recall and precision for Collaborative Filtering.
        random_seed [int]:
            random seed values.
        

            
    RETURN:
    -------------------------------
        the dictionary involve arrayes which are needed to aggrigate MAE, RMSE, recall, precision. 

    
    EXAMPLE:
    -------------------------------
        import numpy as np
        
        # set data
        train_user_ids = np.random.choice(range(20), size=1000)
        train_item_ids = np.random.choice(range(30), size=1000)
        train_values   = np.random.choice(range(6),  size=1000)

        test_user_ids = np.random.choice(range(20), size=500)
        test_item_ids = np.random.choice(range(30), size=500)
        test_values   = np.random.choice(range(6),  size=500)

        # set simple model    
        class random_model:
            def fit(self, user_ids, item_ids, values):
                return self
            def predict(self, user_ids, item_ids):
                return [np.random.rand() for u in user_ids]
    
        model = random_model()        

        result_dict = get_hit_arrays(
                            train_user_ids, train_item_ids, train_values
                           ,test_user_ids, test_item_ids, test_values
                           ,model
                           ,topN=[5,10,15]
                           )
        print(result_dict)
    """
    # --- initial process
    np.random.seed(seed=random_seed)
    
    # ---- Training ----
    model.fit(train_user_ids, train_item_ids, train_values, **fit_args)
        
    # ---- Testing ----
    ## get recall and precision
    good_indexes = [val >= good_score_threshold for val in test_values]
    
    test_good_user_ids = test_user_ids[good_indexes]
    test_good_item_ids = test_item_ids[good_indexes]
    result_of_hit = {N:[] for N in topN}
    
    all_item_ids = set(list(train_item_ids) + list(test_item_ids))
    for user_id, item_id in zip(test_good_user_ids, test_good_item_ids):
        if remove_still_interaction_from_test:
            already_used_item_ids = train_item_ids[train_user_ids == user_id] 
            _all_item_ids = all_item_ids - set(already_used_item_ids)
        else:
            _all_item_ids = deepcopy(all_item_ids)

        # get hit or not
        hit_or_not = get_hit(model, user_id, item_id, _all_item_ids
                             ,topN, n_random_selected_item_ids
                             ,random_seed, **fit_args)
        # count up by each topN
        for N in result_of_hit:
            result_of_hit[N].append(hit_or_not[N])

            
        
    # ---- return metric ----
    sum_result_of_hit = {N:{
            'sum_of_hit' : sum(hit_list),
            'recall' : np.mean(hit_list),
            'precision': np.mean(hit_list)/N,
            } 
            for N,hit_list in result_of_hit.items()}
        
    return sum_result_of_hit


def get_hit(fitted_model, user_id, item_id, all_item_ids
            ,topN=[5,10], n_random_selected_item_ids=1000
            ,random_seed=None, **fit_args):
    """
    get hit or not with in top-N recommendated list.
    details is in (*1).
    
    (*1) Paolo 2010: Performance of Recommender Algorithms on Top-N Recommendation Tasks.
    
    Arguments:
    -------------------------
    fitted_model [fitted recommendation model]:
        this must has a function which is 
        "fitted_model.predict(user_ids, item_ids)"
    
    user_id, item_id:
        the result of this (user_id, item_id) should be good.
    
    all_item_ids [array like object]:
        all item_id
        
    topN [list]:
        list of N.
    
    n_random_selected_item_ids [int]:
        the number of randomly selected from all_item_ids.
    
    
    Return:
    -----------------------
    hit or not of topN respectively. 
    > {5:0, 10:1} # if topN=5 it is not hit, but 10 is hit.
    
    Example:
    -----------------------
    import numpy as np
    
    # set simple model    
    class random_model:
        def fit(self, user_ids, item_ids, values):
            # no training
            return self
        def predict(self, user_ids, item_ids):
            # predict randomly
            return [np.random.rand() for u in user_ids]
    
    fitted_model = random_model().fit(user_ids=[], item_ids=[], values=[])
    
    user_id = '1'
    item_id = '5'
    all_item_ids = ['0','1','2','3','4','6','7','8','9']
    topN=[2,3,4]
        
    result = get_hit(fitted_model, user_id, item_id, all_item_ids, topN=topN) 
    print(result)
    > {2: 0, 3: 1, 4: 1} # result of hit or not of each topN
    
    """
    # --- initial process
    np.random.seed(seed=random_seed)
    
    # remove the item_id from item_id_set
    _all_item_ids = set(all_item_ids) - {item_id}

    # get random selected item_ids as that the user_id will not be interested in. 
    n_random_selected_item_ids = min(n_random_selected_item_ids, len(_all_item_ids))
    random_selected_item_ids = np.random.choice(list(_all_item_ids), size=n_random_selected_item_ids, replace=False)
    
    predicted_random_item_ids = fitted_model.predict([user_id]*n_random_selected_item_ids, random_selected_item_ids, **fit_args)
    predicted_the_item_id = fitted_model.predict([user_id], [item_id], **fit_args)
    
    # join
    predicted = np.concatenate([predicted_random_item_ids, predicted_the_item_id])
    item_ids  = np.concatenate([random_selected_item_ids, [item_id]])
    item_ids__predicted = np.concatenate([item_ids.reshape(-1,1), predicted.reshape(-1,1)], axis=1)
    
    # sort by random shuffle
    np.random.shuffle(item_ids__predicted)
    
    # sort by predicted rating.
    item_ids__predicted = item_ids__predicted[np.argsort(item_ids__predicted[:,1])[::-1], :]
    
    # set the result in test_good_result.
    result = dict()
    for N in topN:
        item_ids_in_topN = item_ids__predicted[:N, 0]                
        if item_id in item_ids_in_topN:
            hit = 1
        else:
            hit = 0    
        result[N] = hit
    
    return result
    
    
    
    
if __name__ == 'This is usage': 

    ###################
    # setup
    
    import numpy as np

    # set simple radom recommendation model    
    class random_model:
        def fit(self, user_ids, item_ids, values):
            # no training
            return self
        def predict(self, user_ids, item_ids):
            # predict randomly
            return [np.random.rand() for u in user_ids]

    ###################
    # get hit=(0 or 1) from a test set (user_id, item_id) and a fitted model.
    from performance_of_topN_recommendation import get_hit
        
    fitted_model = random_model().fit(user_ids=[], item_ids=[], values=[])
    
    user_id = '1'
    item_id = '5'
    all_item_ids = ['0','1','2','3','4','6','7','8','9']
    topN=[2,3,4]
        
    result = get_hit(fitted_model, user_id, item_id, all_item_ids, topN=topN) 
    
    print(result)
    # >{2: 0, 3: 1, 4: 1} # result of hit or not of each topN

    ###################
    # get recall and precision from train and test dataset
    from performance_of_topN_recommendation import get_hit_arrays
    
    # set data
    train_user_ids = np.random.choice(range(20), size=1000)
    train_item_ids = np.random.choice(range(30), size=1000)
    train_values   = np.random.choice(range(6),  size=1000)

    test_user_ids = np.random.choice(range(20), size=500)
    test_item_ids = np.random.choice(range(30), size=500)
    test_values   = np.random.choice(range(6),  size=500)

    # set simple model    
    class random_model:
        def fit(self, user_ids, item_ids, values):
            return self
        def predict(self, user_ids, item_ids):
            return [np.random.rand() for u in user_ids]

    model = random_model()        

    result_dict = get_hit_arrays(
                        train_user_ids, train_item_ids, train_values
                       ,test_user_ids, test_item_ids, test_values
                       ,model
                       ,topN=[5,10,15]
                       )
    print(result_dict)
    """ > {
    5: {'sum_of_hit': 20, 'recall': 0.23529411764705882, 'precision': 0.047058823529411764}, 
    10: {'sum_of_hit': 32, 'recall': 0.3764705882352941, 'precision': 0.03764705882352941}, 
    15: {'sum_of_hit': 40, 'recall': 0.47058823529411764, 'precision': 0.03137254901960784}
    }
    """
    

