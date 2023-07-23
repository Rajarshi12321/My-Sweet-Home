import numpy as np
import pandas as pd
# from pyodide.http import open_url

url = "artifact/Dataset.csv"
# url = "https://github.com/Rajarshi12321/Housing_predict_recommend/blob/main/artifact/Dataset.csv"
# url_content = open_url(url)

# dataset= pd.read_csv(url)
path = "city_locality.npy"

city_loc =np.load(path)

def main_fun():
    print(city_loc[0][0])
    return city_loc[0][0]

def city_arr():
    city_set = set()
    for ele in city_loc:
        city_set.add(ele[0])

    return city_set

def loc_arr(city):
    loc_set = set()
    for i in city_loc:
        if i[0] == city:
            loc_set.add(i[1])
            

    return loc_set

# main_fun()
