from src.recommender.data_transformation_recommend import DataTransformationRecommend
import os
import sys
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
logging.info("no , src.utils")


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class Recommender:
    def __init__(self):
        pass

    def get_similar_houses(propType, loc, furn, fn, Fnum, city, bed, Bath, RorS, PostDays, dataset, n=6):

        # converting the text data to feature vectors

        vectorize = TfidfVectorizer()

        # making input str
        input_str = propType + " " + loc + " " + furn + " " + fn + " " + Fnum + \
            " " + city + " " + bed + " " + Bath + " " + RorS + " " + PostDays

        # Data_path = "artifacts\recommend.csv"
        # dataset = pd.read_csv(Data_path)

        # adding the input for tfdif
        dataset.loc[len(dataset.index)] = input_str

        # vecterization
        feature_vectors = vectorize.fit_transform(dataset["text"])

        # finding cosine similarity
        dataset['distances'] = cosine_similarity(
            feature_vectors, [feature_vectors.toarray()[-1]])

        # this contains the parent itself as the most similar entry, hence n+1 to get n children
        n_largest = dataset['distances'].nlargest(n + 2)

        # Return the top similar houses
        return dataset.loc[n_largest.index[2:]]


# if __name__ == "__main__":

#     # Getting the data for processing
#     data_transform = DataTransformationRecommend()

#     data, _ = data_transform.initiate_data_transformation_recommend()
#     # print(data)

#     recommend = Recommender
#     similar_houses = recommend.get_similar_houses(
#         'Multistorey Apartm',  'Danapur',  'Semi-Furnished',  '', '6', 'Patna',  '0', '0',   'Sale', '', data, n=6)

#     print(similar_houses[["city", "URLs", "RentOrSale",
#           "furnishing", "locality", "propertyType"]])
