from lenskit.algorithms import Recommender  
from lenskit.algorithms.item_knn import ItemItem 







class Individual_Recommenders: 

    
    def useruser_individual_recommender_train(self, train_df):  
        item_item = ItemItem(15, min_nbrs=3)  # Minimum (3) and maximum (15) number of neighbors to consider  
        recsys = Recommender.adapt(item_item)  
        recsys.fit(train_df)  
        return recsys  