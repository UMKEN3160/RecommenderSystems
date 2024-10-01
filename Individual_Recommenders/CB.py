import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, ratings, wines):
        self.ratings = ratings
        self.wines = wines
        self.similarity_matrix = None

    def fit(self):
        # create feature matrix (e.g., one-hot encode categorical features)
        features = pd.get_dummies(self.wines[['Type', 'Country', 'Body']])
        # compute the cosine similarity matrix
        self.similarity_matrix = cosine_similarity(features)

    def recommend(self, user_id, top_n=5):
        # get wines rated by the user
        user_wines = self.ratings[self.ratings['UserID'] == user_id]
        if user_wines.empty:
            return []
        
        # get the indices of the wines the user has rated
        rated_indices = user_wines['WineID'].values
        
        # compute similarity scores for rated wines
        scores = pd.Series(0, index=self.wines['WineID'])
        for wine_id in rated_indices:
            index = self.wines.index[self.wines['WineID'] == wine_id].tolist()[0]
            similar_scores = self.similarity_matrix[index]
            scores += similar_scores
            
        # exclude wines already rated by the user
        scores = scores[~scores.index.isin(rated_indices)]
        
        # get the top 5 recommendations
        recommendations = scores.nlargest(top_n).index.tolist()
        return recommendations

