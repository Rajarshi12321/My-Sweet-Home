import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
from src.recommender.house_recommender import Recommender


class PredictRecommendPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            fea_df = pd.DataFrame(features, columns=['propertyType', 'locality', 'furnishing',
                                                     'city', 'bedrooms', 'bathrooms', 'RentOrSale',  'exactPrice'])
            data_scaled = preprocessor.transform(fea_df)
            preds = model.predict(data_scaled[:, :-1])
            prediction = round(np.exp(preds[0]))
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

    def recommend(self, features):
        try:
            Data_path = "artifacts/recommend_data.csv"
            data = pd.read_csv(Data_path)

            recommend = Recommender
            similar_houses = recommend.get_similar_houses(
                features[0],   features[1],   features[2],  features[3],   features[4],  features[5],    features[6],
                dataset=data)

            return similar_houses

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 propertyType: str,
                 locality: str,
                 furnishing: str,
                 city: str,
                 bedrooms: str,
                 bathrooms: str,
                 RentOrSale: str):

        self.propertyType = propertyType

        self.locality = locality

        self.furnishing = furnishing

        self.city = city

        self.bedrooms = bedrooms

        self.bathrooms = bathrooms

        self.RentOrSale = RentOrSale

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "propertyType": [self.propertyType],
                "locality": [self.locality],
                "furnishing": [self.furnishing],
                "city": [self.city],
                "bedrooms": [self.bedrooms],
                "bathrooms": [self.bathrooms],
                "RentOrSale": [self.RentOrSale],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
