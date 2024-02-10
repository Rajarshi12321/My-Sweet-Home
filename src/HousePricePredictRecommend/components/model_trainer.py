import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# from joblib import dump

from HousePricePredictRecommend.utils.exception import CustomException
from HousePricePredictRecommend import logging

from HousePricePredictRecommend.utils.common import save_object, evaluate_models, remove_outliers_iqr
from HousePricePredictRecommend.entity.config_entity import TrainingConfig
from dotenv import load_dotenv
import os

# This try catch block will run only while model training experiments not on production deployment
try:
    # Load environment variables from .env file
    load_dotenv()

    # Access environment variables
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
    MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
except:
    pass


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def initiate_model_trainer(self):
        DF = pd.read_csv(self.config.training_data)

        imp_feature = ['propertyType',
                       'locality',
                       'furnishing',
                       'city',
                       'bedrooms',
                       'bathrooms',
                       'RentOrSale',]

        DF = remove_outliers_iqr(DF)
        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = train_test_split(
                DF[imp_feature], DF[["exactPrice"]], test_size=0.33, random_state=42)

            print(y_train)
            # To convert y_train to required format
            y_train = np.ravel(y_train)
            print(y_train)

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),

                "AdaBoostRegressor": AdaBoostRegressor(),
            }
            print("OKK 1")

            params = self.config.model_params
            print(params.MODELS)
            print("OKK 2")
            models = {key: value for key,
                      value in models.items() if key in params.MODELS}
            print(models)

            print("OKK 3")

            model_report, parameters = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                       models=models, param=params.MODELS)
            print("OKK 4")
            logging.info(f"models -> {model_report}")

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best found model on both training and testing dataset")

            # Saving model in artifacts, which is not being tracked
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            # Saving model in model folder, which is being tracked for the predection pipeline
            save_object(
                file_path=os.path.join("model", "model.pkl"),
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            print(mae)

            # Log into MLflow
            self.log_model_into_mlflow(
                model_report, parameters, y_test, predicted, "Full Data Model")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer_rent(self):
        DF = pd.read_csv(self.config.training_data)

        imp_feature = ['propertyType',
                       'locality',
                       'furnishing',
                       'city',
                       'bedrooms',
                       'bathrooms',
                       'RentOrSale',]

        DF = DF[DF["RentOrSale"] == 1]

        DF = remove_outliers_iqr(DF)
        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = train_test_split(
                DF[imp_feature], DF[["exactPrice"]], test_size=0.33, random_state=42)

            print(y_train)
            # To convert y_train to required format
            y_train = np.ravel(y_train)
            print(y_train)

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),

                "AdaBoostRegressor": AdaBoostRegressor(),
            }
            print("OKK 1")

            params = self.config.model_params
            print(params.MODELS)
            print("OKK 2")
            models = {key: value for key,
                      value in models.items() if key in params.MODELS}
            print(models)

            print("OKK 3")

            model_report, parameters = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                       models=models, param=params.MODELS)
            print("OKK 4")
            logging.info(f"models -> {model_report}")

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.3:
                raise CustomException("No best model found")
            logging.info(
                f"Best found model on both training and testing dataset")

            # Saving model in artifacts, which is not being tracked
            save_object(
                file_path=self.config.trained_model_file_path_rent,
                obj=best_model
            )

            # Saving model in model folder, which is being tracked for the predection pipeline

            save_object(
                file_path=os.path.join("model", "model_rent.pkl"),
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            print(mae)

            # Log into MLflow
            # with mlflow.start_run(run_name="Rent Data Model"):
            #     for model_name, model in model_report.items():
            #         mlflow.log_params(parameters)  # Log model parameters
            #         mlflow.log_metric("mean_squared_error", mean_squared_error(y_test, predicted))  # Log MSE
            #         mlflow.log_metric("mean_absolute_error", mean_absolute_error(y_test, predicted))  # Log MSE
            #         mlflow.log_metric("r2_score", r2_score(y_test, predicted))  # Log R2 Score
            #         mlflow.sklearn.log_model(model, f"{model_name}_model")  # Log the trained model

            #         # Model registry does not work with file store
            #         if tracking_url_type_store != "file":

            #             # Register the model
            #             # There are other ways to use the Model Registry, which depends on the use case,
            #             # please refer to the doc for more information:
            #             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #             mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            #         else:
            #             mlflow.keras.log_model(self.model, "model")

            self.log_model_into_mlflow(
                model_report, parameters, y_test, predicted, "Rent Data Model")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)

    def log_model_into_mlflow(self, model_report, parameters, y_test, predicted, run_name):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=run_name):
            for model_name, model in model_report.items():
                mlflow.log_params(parameters)  # Log model parameters
                mlflow.log_metric("mean_squared_error", mean_squared_error(
                    y_test, predicted))  # Log MSE
                mlflow.log_metric("mean_absolute_error", mean_absolute_error(
                    y_test, predicted))  # Log MSE
                mlflow.log_metric("r2_score", r2_score(
                    y_test, predicted))  # Log R2 Score

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(
                        model, f"{model_name}_model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
