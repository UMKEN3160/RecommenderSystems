
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class ContentBasedRecommender:
    def __init__(self, ratings, wines):
        self.ratings = ratings
<<<<<<< HEAD
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
=======
        self.wines = wines
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()  

    def extract_features(self, ratings):
        
        # Combine ratings with wine data to get features
        merged_data = ratings.merge(self.wines, on='WineID')
        
        # Use OneHotEncoder for categorical variables
        features = merged_data[['Type', 'Country', 'Body']]
        
        # Fit the encoder on the training data only
        self.encoder.fit(features)
        
        # Transform features
        return self.encoder.transform(features)

    def fit(self, train_ratings):
        X_train = self.extract_features(train_ratings) 
        y_train = train_ratings['Rating']  

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)

        # Fit the SVR model
        self.model = SVR(kernel='linear', C=1.0, epsilon=0.1, gamma='scale')
        self.model.fit(X_train, y_train)

    def recommend(self, user_id, top_n=5):
        # Wines user has already rated
        user_ratings = self.ratings[self.ratings['UserID'] == user_id]
        rated_wines = user_ratings['WineID'].values
>>>>>>> 24a28ffb67be08e6b70947685c55b311a7bcffe8

        # Unrated wines
        unrated_wines = self.wines[~self.wines['WineID'].isin(rated_wines)]
        
        if unrated_wines.empty:
            return []

        # Prepare features for unrated wines
        X_unrated = self.encoder.transform(unrated_wines[['Type', 'Country', 'Body']])
        X_unrated = self.scaler.transform(X_unrated)  

        predicted_ratings = self.model.predict(X_unrated)

        unrated_wines['PredictedRating'] = predicted_ratings

        recommendations = unrated_wines.nlargest(top_n, 'PredictedRating')['WineID'].tolist()

        return recommendations
