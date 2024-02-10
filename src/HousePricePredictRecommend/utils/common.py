from HousePricePredictRecommend import logging
from HousePricePredictRecommend.utils.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle
import dill
import pandas as pd
import numpy as np
import datetime
import sys
import os
from box.exceptions import BoxValueError
import yaml
from HousePricePredictRecommend import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


##########


def date_transform(date):
    try:
        date_object = datetime.datetime.strptime(date, "%b %d, '%y")
        formatted_date = datetime.datetime.strftime(date_object, "%Y-%m-%d")
        new_date = datetime.datetime.strptime(
            formatted_date, "%Y-%m-%d").date()
        current_date = datetime.date.today()
        Days = (current_date - new_date).days
        return Days

    except Exception as e:
        raise CustomException(e, sys)


def remove_outliers_iqr(df):
    # Calculate the first quartile (25th percentile) and third quartile (75th percentile)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if lower_bound['exactPrice'] < 0:
        lower_bound = df.min()

    # Filter rows where any value is within the lower and upper bounds
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    return df_no_outliers


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        print(models)
        print(param, "utils")

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        logging.info(f"utils model trainer : {report} , { model.get_params()}")
        return report, model.get_params()

    except Exception as e:
        raise CustomException(e, sys)


class ReplaceValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, value, replace_with):
        self.value = value
        self.replace_with = replace_with

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_replaced = X.replace(self.value, self.replace_with)
        return X_replaced


class DropNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, subset):
        self.subset = subset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_dropped = X.dropna(subset=self.subset)
        return X_dropped


class CategoricalLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        # logging.info(f"Debugging- feature1 : {categorical_features}")

        if "exactPrice" in categorical_features:
            categorical_features.remove('exactPrice')
        # logging.info(f"Debugging- feature2 : {categorical_features}")
        self.categorical_features = categorical_features
        self.labels_ordered_dict = {}

    def fit(self, X, y=None):
        for feature in self.categorical_features:
            labels_ordered = X.groupby(
                [feature])['exactPrice'].mean().sort_values().index
            labels_ordered_dict = {k: i for i,
                                   k in enumerate(labels_ordered, 0)}
            self.labels_ordered_dict[feature] = labels_ordered_dict
        return self

    def transform(self, X):
        X_transformed = X.copy()
        # logging.info(f"Debugging- feature3 : {X.columns}")

        for feature in X.columns:
            if feature != 'exactPrice' and feature != 'postedOn':
                X_transformed[feature] = X_transformed[feature].map(
                    self.labels_ordered_dict[feature])
        return X_transformed


class DateTransformTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.date_column] = X_transformed[self.date_column].apply(
            self.date_transform)
        return X_transformed

    @staticmethod
    def date_transform(date):
        date_object = datetime.datetime.strptime(date, "%b %d, '%y")
        formatted_date = datetime.datetime.strftime(date_object, "%Y-%m-%d")
        new_date = datetime.datetime.strptime(
            formatted_date, "%Y-%m-%d").date()
        current_date = datetime.date.today()
        ab = (current_date - new_date).days
        return ab


class FillnaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, value):
        self.columns = columns
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = X_transformed[self.columns].fillna(
            value=self.value)
        return X_transformed


def format_indian_currency(num, currency_symbol='₹'):
    # Convert the number to a string and reverse it for easier processing
    num_str = str(num)[::-1]

    # Create a list to store the formatted parts of the number

    formatted_parts = []

    chunk = num_str[0:3][::-1]
    formatted_parts.append(chunk)
    # Iterate through the reversed number string in chunks of three digits
    for i in range(3, len(num_str), 2):
        # Get the chunk and reverse it back to the original order
        chunk = num_str[i:i+2][::-1]
        formatted_parts.append(chunk)

    # Reverse the list to get the correct order and join the parts with commas
    formatted_number = ','.join(formatted_parts[::-1])

    # Add the currency symbol at the beginning of the formatted number
    formatted_number = f"{currency_symbol}{formatted_number}"

    return formatted_number


def output_within_range(number):
    # Calculate the output within the range of 15000 to 20000
    number1 = number/pow(10, 3)
    number1 = np.round(number1) * pow(10, 3)
    number2 = number/pow(10, 4)
    number2 = np.round(number2) * pow(10, 4)
    # Convert the output to a string and return it
    num1 = format_indian_currency(str(number1)[:-2])
    num2 = format_indian_currency(str(number2)[:-2])

    if number1 >= number2:
        output_str = f"{num2} - {num1}"
    else:

        output_str = f"{num1} - {num2}"

    print(number)
    # print(format_indian_currency(str(number1)[:-2]))
    return output_str
