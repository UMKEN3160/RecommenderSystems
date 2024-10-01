


from lenskit.algorithms import Recommender  
from lenskit.algorithms.item_knn import ItemItem 
# from lenskit.similarity import CosineSim
# from lenskit.algorithms.fallback import Fallback
from lenskit.algorithms.basic import Bias
##standard imports
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix as csr
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from matplotlib import pyplot as plt
import cv2 as cv
import scipy.sparse as sp
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import mean_average_precision_at_k


warnings.simplefilter(action='ignore', category=FutureWarning)



class Individual_Recommenders: 

    
    def useruser_individual_recommender_train(self, train_df):  

        item_item = ItemItem(15, min_nbrs=3)  # Minimum (3) and maximum (15) number of neighbors to consider  
        ##recsys = Recommender.adapt(item_item)  
        
        # bias_predictor = Bias()  
        recsys=Recommender.adapt(item_item)
        recsys.fit(train_df)  

        return recsys  

#lets make a class for the modelss
class contentBased:
    
    def __init__(self,user_reviews,wine_data):
        self.user_reviews=user_reviews
        # Preprocessing for 'Grapes', 'Harmonize', and 'Vintages'
        print(wine_data['Grapes'])
        wine_data['Grapes'] = wine_data['Grapes'].apply(lambda x: ' '.join(eval(x)))  # Convert list to space-separated string
        wine_data['Harmonize'] = wine_data['Harmonize'].apply(lambda x: ' '.join(eval(x)))




        # Function to handle mixed vintages: list or integer
        def convert_to_list(vintages):
            if isinstance(vintages, int):
                # If the vintage is a single integer, wrap it in a list
                return [str(vintages)]
            elif isinstance(vintages, str):
                # If it's a string, evaluate it as a list (since strings in CSV could be saved as "['1990', '1995']")
                try:
                    return list(map(str, eval(vintages)))  # Convert all elements to strings
                except:
                    return [str(vintages)]  # In case it's a string number
            else:
                # In case it's an unexpected format, treat as string
                return [str(vintages)]

        # Apply the conversion to the 'Vintages' column
        wine_data['Vintages'] = wine_data['Vintages'].apply(convert_to_list)

        # Convert the list into a space-separated string
        wine_data['Vintages'] = wine_data['Vintages'].apply(lambda x: ' '.join(x))

      
        
        tfidf = TfidfVectorizer()
       
        wine_data_tfidf_grapes = tfidf.fit_transform(wine_data['Grapes'])
        wine_data_tfidf_harmonize = tfidf.fit_transform(wine_data['Harmonize'])
        wine_data_tfidf_vintages = tfidf.fit_transform(wine_data['Vintages'])

        # One-hot encoding for 'Type', 'Body', 'Acidity', 'Country', 'RegionName', 'WineryName'
        one_hot = OneHotEncoder()

        categorical_features = ['Type', 'Body', 'Acidity', 'Country', 'RegionName', 'WineryName']
        wine_categorical_onehot = one_hot.fit_transform(wine_data[categorical_features])

        # Scale the numerical feature 'ABV'
        scaler = StandardScaler()
        wine_abv_scaled = scaler.fit_transform(wine_data[['ABV']])

        # Define the weights for different features
        weights ={'Grapes' : 2.6351178844772396, 'Harmonize' : 1.553630179215512, 'Vintages' : 1.760447782530082, 'Type' : 2.5859874914949206, 'Body' : 1.2628975201148853, 'Acidity' : 0.9828619145406399, 'Country' : 0.6956261879697497, 'Region' : 0.6957519257816862, 'Winery' : 0.7668374998938243, 'ABV' : 2.11521395128059}
        # Apply weights to categorical features (by columns)
        wine_type_onehot = wine_categorical_onehot[:, :one_hot.categories_[0].size] * weights['Type']  # 'Type' weight
        wine_body_onehot = wine_categorical_onehot[:, one_hot.categories_[0].size:one_hot.categories_[0].size + one_hot.categories_[1].size] * weights['Body']
        wine_acidity_onehot = wine_categorical_onehot[:, one_hot.categories_[0].size + one_hot.categories_[1].size:one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size] * weights['Acidity']
        wine_country_onehot = wine_categorical_onehot[:, one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size:one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size + one_hot.categories_[3].size] * weights['Country']
        wine_region_onehot = wine_categorical_onehot[:, one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size + one_hot.categories_[3].size:one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size + one_hot.categories_[3].size + one_hot.categories_[4].size] * weights['Region']
        wine_winery_onehot = wine_categorical_onehot[:, one_hot.categories_[0].size + one_hot.categories_[1].size + one_hot.categories_[2].size + one_hot.categories_[3].size + one_hot.categories_[4].size:] * weights['Winery']

        # Scale the numerical feature 'ABV'
        scaler = StandardScaler()
        wine_abv_scaled = scaler.fit_transform(wine_data[['ABV']]) * weights['ABV']  # Apply weight for 'ABV'
        wine_data_tfidf_grapes = wine_data_tfidf_grapes * weights['Grapes']  # Apply weight for 'Grapes'
        wine_data_tfidf_harmonize = wine_data_tfidf_harmonize * weights['Harmonize']  # Apply weight for 'Harmonize'
        wine_data_tfidf_vintages = wine_data_tfidf_vintages * weights['Vintages']  # Apply weight for 'Vintages'
        
        wine_features = sp.hstack([
            wine_data_tfidf_grapes, 
            wine_data_tfidf_harmonize, 
            wine_data_tfidf_vintages,
            wine_type_onehot, 
            wine_body_onehot, 
            wine_acidity_onehot,
            wine_country_onehot, 
            wine_region_onehot, 
            wine_winery_onehot, 
            wine_abv_scaled
        ])
                # Compute cosine similarity between all wines
        wine_similarity = cosine_similarity(wine_features)

        # Store this in a DataFrame for easier access
        wine_similarity_df = pd.DataFrame(wine_similarity, index=wine_data['WineID'], columns=wine_data['WineID'])

        self.wine_similarity_df=wine_similarity_df

    def recommend_wines(self,user_id, top_n=5,returnall=False):
        # Get the wines the user has rated
        user_wines = self.user_reviews[self.user_reviews['UserID'] == user_id]
        # Filter for wines the user rated 4 or 5
        high_rated_wines = user_wines[user_wines['Rating'] >= 4]['WineID']
        # Get similarity scores for each of these wines
        similar_wines = pd.Series(dtype=float)
        for wine_id in high_rated_wines:
            similar_wines = similar_wines.append(self.wine_similarity_df[wine_id])
        # Sort the wines by similarity score
        similar_wines = similar_wines.groupby(similar_wines.index).mean().sort_values(ascending=False)
        # Exclude wines the user has already rated
        similar_wines = similar_wines[~similar_wines.index.isin(user_wines['WineID'])]
        if returnall:
            return similar_wines.index.tolist()
        return similar_wines.head(top_n).index.tolist()

class collaborativeFiltering:
    def __init__(self,user_reviews):
        self.user_reviews=user_reviews
        self.df_filtered = self.user_reviews[['UserID', 'WineID', 'Rating']]
        self.user_item_matrix = csr_matrix((self.df_filtered['Rating'], (self.df_filtered['UserID'], self.df_filtered['WineID'])))
        self.model = AlternatingLeastSquares(factors=100, regularization=0.01, iterations=50)
        self.model.fit(self.user_item_matrix)
    def recommend_wines(self,user_id, top_n=5,returnall=False):
        # Get the user's ratings
        user_ratings = self.df_filtered[self.df_filtered['UserID'] == user_id]
        # Predict ratings for all wines
        user_wine_scores = self.model.recommend(user_id, self.user_item_matrix[user_id], N=10)
        if returnall:
            return user_wine_scores
        return user_wine_scores.head(top_n).index.tolist()

class hybridModel:
    def __init__(self,user_reviews,wine_data,ratio):
        self.user_reviews=user_reviews
        self.wine_data=wine_data
        self.contentBasedModel=contentBased(self.user_reviews,self.wine_data)
        self.collaborativeFilteringModel=collaborativeFiltering(self.user_reviews)
        self.ratio=ratio	
    def recommend_wines(self,user_id, top_n=5,returnall=False):
        contentBasedWines=self.contentBasedModel.recommend_wines(user_id, top_n,returnall)
        collaborativeFilteringWines=self.collaborativeFilteringModel.recommend_wines(user_id, top_n,returnall)
        allWines=self.ratio*contentBasedWines+(1-self.ratio)*collaborativeFilteringWines
        return allWines

    

                                                                  