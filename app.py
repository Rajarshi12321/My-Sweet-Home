# main.py
import numpy as np
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

# Load city_locality data
path = "city_locality.npy"
city_loc = np.load(path)

# Function to get unique cities


def city_arr():
    city_set = set()
    for ele in city_loc:
        city_set.add(ele[0])
    return city_set

# Function to get localities for a given city


def main_arr(city):
    loc_set = set()
    for i in city_loc:
        if i[0] == city:
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


@app.route('/api/main_arr/<selected_prop_type>', methods=['GET'])
def get_main_arr(selected_city):
    localities = main_arr(selected_city)
    return jsonify(localities)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
