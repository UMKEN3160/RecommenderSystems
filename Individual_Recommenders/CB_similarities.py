import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class CBRecommender:
    def __init__(self, ratings, wines):
        self.ratings = ratings
        self.wines = wines
        self.similarity_matrix = None

    def fit(self, train_ratings):
        # Create feature matrix (e.g., one-hot encode categorical features)
        features = pd.get_dummies(self.wines[['Type', 'Country', 'Body']])
        # Compute the cosine similarity matrix based on wine features
        self.similarity_matrix = cosine_similarity(features)
        # Set the training ratings as an attribute to filter users' ratings
        self.train_ratings = train_ratings

    def recommend(self, user_id, top_n=5):
        # Get wines rated by the user in the training set
        user_wines = self.train_ratings[self.train_ratings['UserID'] == user_id]
        if user_wines.empty:
            return []
        
        # Get the indices of the wines the user has rated
        rated_indices = user_wines['WineID'].values
        
        # Compute similarity scores for rated wines
        scores = pd.Series(0, index=self.wines['WineID'])
        for wine_id in rated_indices:
            index = self.wines.index[self.wines['WineID'] == wine_id].tolist()[0]
            similar_scores = self.similarity_matrix[index]
            scores += similar_scores
            
        # Exclude wines already rated by the user
        scores = scores[~scores.index.isin(rated_indices)]
        
        # Get the top_n recommendations
        recommendations = scores.nlargest(top_n).index.tolist()
        return recommendations

