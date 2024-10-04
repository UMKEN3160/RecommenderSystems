import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class CBRecommender:
    def __init__(self, ratings, wines):
        self.ratings = ratings
        self.wines = wines.reset_index(drop=True)
        self.similarity_matrix = None
        self.wine_id_to_index = None
    def fit(self):
        # create feature matrix (e.g., one-hot encode categorical features)
        features = pd.get_dummies(self.wines[['Type', 'Country', 'Body']])
        # compute the cosine similarity matrix
        self.similarity_matrix = cosine_similarity(features)
        # Create a mapping from WineID to index
        self.wine_id_to_index = pd.Series(self.wines.index, self.wines['WineID']).to_dict()
    def recommend(self, user_id, top_n=5):
        # get wines rated by the user
        user_wines = self.ratings[self.ratings['UserID'] == user_id]
        if user_wines.empty:
            print("user has rated no wines")
            return []
        
        # get the indices of the wines the user has rated
        rated_indices = user_wines['WineID'].values
        
        # compute similarity scores for rated wines
        scores = pd.Series(0, index=self.wines['WineID'])
        for wine_id in rated_indices:
            if wine_id in self.wines['WineID'].values:
                index = self.wine_id_to_index[wine_id]
                similar_scores = self.similarity_matrix[index]
                scores += similar_scores
            else:
                print("WineID not in wine dataset")
        # exclude wines already rated by the user
        scores = scores[~scores.index.isin(rated_indices)]
        
        # get the top 5 recommendations
        recommendations = scores.nlargest(top_n).index.tolist()
        return recommendations

