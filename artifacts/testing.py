from src.utils import load_object, date_transform
from src.recommender.house_recommender import Recommender

import pandas as pd
import numpy as np
import math


model = load_object("artifacts\model.pkl")

print(model.get_params())
print(model)


pro = load_object("artifacts\preprocessor.pkl")


fea = [["Residential House", "Phase 1 Ashiana Nagar", "Semi-Furnished",
        "Ground", 2.0, "Patna", 3.0, 3.0, "Rent", "Jun 21, '23", 17000.0]]
# Convert fea to a DataFrame
fea_df = pd.DataFrame(fea, columns=['propertyType', 'locality', 'furnishing', 'flrNum',
                      'totalFlrNum', 'city', 'bedrooms', 'bathrooms', 'RentOrSale', 'postedOn', 'exactPrice'])

data_preprocess = pro.transform(fea_df)
preds = model.predict(data_preprocess[:, :-1])

print(preds)
# print(np.exp(preds))
print(round(np.exp(preds[0])))

# Recommender
Data_path = "artifacts\recommend_data.csv"
data = pd.read_csv(Data_path)

recommend = Recommender
similar_houses = recommend.get_similar_houses(
    "Residential House", "Phase 1 Ashiana Nagar", "Semi-Furnished",
    "Ground", "2.0", "Patna", "3.0", "3.0", "Rent", "Jun 21, '23", dataset=data)

print(similar_houses)

print(similar_houses[["URLs"]])
