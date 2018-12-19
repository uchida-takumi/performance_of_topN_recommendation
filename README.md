# performance_of_topN_recommendation
This is my implementation of [Paolo 2010: Performance of Recommender Algorithms on Top-N Recommendation Tasks]

# how to use

```
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

```



# Reference:
```
@inproceedings{cremonesi2010performance,
  title={Performance of recommender algorithms on top-n recommendation tasks},
  author={Cremonesi, Paolo and Koren, Yehuda and Turrin, Roberto},
  booktitle={Proceedings of the fourth ACM conference on Recommender systems},
  pages={39--46},
  year={2010},
  organization={ACM}
}
```