# main.py
import numpy as np
from flask import Flask, jsonify, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictRecommendPipeline
from src.pipeline.scraping_pipeline import ImageScrappingPipeline
from math import trunc

app = Flask(__name__)

# Load city_locality data
path = "city_locality.npy"
city_loc = np.load(path)

# Function to get unique cities

propType = ['Multistorey Apartment', 'Residential House',
            'Builder Floor Apartment', 'Villa', 'Studio Apartment',
            'Penthouse']

RoS = ['Rent', 'Sale']

BHK = ["1", "2", "3", "4", "5", "5+"]

Furnishing = ['Semi-Furnished', 'Furnished', 'Unfurnished']


def city_arr():
    city_set = set()
    for ele in city_loc:
        city_set.add(ele[0])

    return city_set

# Function to get localities for a given city


def main_arr(city):
    loc_set = set()
    for i in city_loc:
        if i[0] == city and i[1] != "Missing":
            loc_set.add(i[1])
    return list(loc_set)


@app.route('/api/city_arr', methods=['GET'])
def get_city_arr():
    cities = city_arr()
    return jsonify(list(cities))


@app.route('/api/main_arr/<selected_city>', methods=['GET'])
def get_main_arr(selected_city):
    localities = main_arr(selected_city)
    return jsonify(localities)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        return render_template('index.html', PropType=propType, BHK=BHK, Furnish=Furnishing, Ros=RoS)

    else:
        print("submitted")
        if request.form.get('BHK') != "5+":
            bed = request.form.get('BHK')
            bath = request.form.get('BHK')
        else:
            bed = "7"
            bath = "7"

        data = CustomData(
            propertyType=request.form.get('propertyType'),
            locality=request.form.get('locality'),
            furnishing=request.form.get('furnishing'),
            city=request.form.get('city'),
            bedrooms=bed,
            bathrooms=bath,
            RentOrSale=request.form.get('RentOrSale'),
            exactPrice=" "

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        # print("Before Prediction")

        predict_recommend_pipeline = PredictRecommendPipeline()
        # print("Mid Prediction")
        result = predict_recommend_pipeline.predict(pred_df)
        # print("after Prediction")
        print(pred_df, "df")
        print(result, "res")
        recommend = predict_recommend_pipeline.recommend(pred_df)

        print(pred_df['propertyType'], "Yupp")
        print(pred_df['locality'], "Yupp")
        print((recommend["distances"].mean())*100, "Yupp")
        similarity = (recommend["distances"].mean())*100
        similarity = trunc(similarity)
        similarity = str(similarity)+"%"

        # img_pipeline = ImageScrappingPipeline
        # recommend = img_pipeline.get_images(recommend)

        # print(recommend)

        return render_template('home.html', PropType=propType, BHK=BHK, Furnish=Furnishing, Ros=RoS, result=result, dataset=recommend, similar=similarity)


# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     return render_template('index.html', PropType=propType, BHK=BHK, Furnish=Furnishing, Ros=RoS)


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=True, debug=True)
