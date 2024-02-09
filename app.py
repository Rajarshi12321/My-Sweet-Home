# app.py
import json
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

from HousePricePredictRecommend.pipeline.predict_recommend_pipeline import CustomData, PredictRecommendPipeline
# from HousePricePredictRecommend.pipeline.scraping_pipeline import ImageScrappingPipeline
from math import trunc
from HousePricePredictRecommend import logging
import os

app = Flask(__name__)
CORS(app)

# # Load city_locality data
# path = "city_locality.npy"
# city_loc = np.load(path, allow_pickle=True)


# Load the JSON data from city_loc.json
with open('city_loc.json', 'r') as json_file:
    city_loc = json.load(json_file)


# Function to get unique cities

propType = ['Multistorey Apartment', 'Residential House',
            'Builder Floor Apartment', 'Villa', 'Studio Apartment',
            'Penthouse']

RoS = ['Rent', 'Sale']

BHK = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

Furnishing = ['Semi-Furnished', 'Furnished', 'Unfurnished']


def city_arr():
    # Retrieve the city names as an array directly from the keys of the JSON data
    city_names = list(city_loc.keys())
    return city_names


# Function to get localities for a given city


# def main_arr(city):
#     loc_set = set()
#     for i in city_loc:
#         if i[0] == city and i[1] != "Missing":
#             loc_set.add(i[1])
#     return list(loc_set)


# Function to get localities for a given city
def main_arr(city):
    loc_set = set()
    # Check if the city exists in the JSON data
    if city in city_loc:
        # Iterate through the localities of the given city
        for place in city_loc[city]:
            if place != "Missing":
                loc_set.add(place)
    return list(loc_set)


@app.route('/api/city_arr', methods=['GET'])
@cross_origin()
def get_city_arr():
    cities = city_arr()
    return jsonify(list(cities))


@app.route('/api/main_arr/<selected_city>', methods=['GET'])
@cross_origin()
def get_main_arr(selected_city):
    localities = main_arr(selected_city)
    return jsonify(localities)


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def home():
    if request.method == "GET":
        return render_template('index.html', PropType=propType, BHK=BHK, Furnish=Furnishing, Ros=RoS)

    else:
        print("submitted")

        data = CustomData(
            propertyType=request.form.get('propertyType'),
            locality=request.form.get('locality'),
            furnishing=request.form.get('furnishing'),
            city=request.form.get('city'),
            bedrooms=request.form.get('BHK'),
            bathrooms=request.form.get('BHK'),
            RentOrSale=request.form.get('RentOrSale'),
            exactPrice=" "

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        print("Before Prediction")

        predict_recommend_pipeline = PredictRecommendPipeline()

        # print("Mid Prediction")

        result = predict_recommend_pipeline.predict(pred_df)

        # logging.info(f"{result} Prediction Result")

        print("after Prediction")
        print(pred_df, "DataFrame")
        print(result, "result")

        recommend = predict_recommend_pipeline.recommend(pred_df)

        similarity = (recommend["distances"].mean())*100
        similarity = trunc(similarity)
        similarity = str(similarity)+"%"

        # img_pipeline = ImageScrappingPipeline
        # recommend = img_pipeline.get_images(recommend)

        # logging.info(
        #     f" {recommend} Recommended properties with {similarity} % similarity")

        print(recommend)

        return render_template('home.html', PropType=propType, BHK=BHK, Furnish=Furnishing, Ros=RoS, result=result, dataset=recommend, similar=similarity)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8080)
