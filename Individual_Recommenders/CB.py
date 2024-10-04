
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

    def fit(self, train_ratings, train_features):
        self.ratings = train_ratings  # Store ratings for later use
        X_train = train_features  # Use the features passed in
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
