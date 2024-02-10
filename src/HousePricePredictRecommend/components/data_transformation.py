import sys
import joblib
from HousePricePredictRecommend.entity.config_entity import DataTransformationConfig
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from HousePricePredictRecommend.utils.exception import CustomException
from HousePricePredictRecommend import logging
from HousePricePredictRecommend.utils.common import DropNaTransformer, FillnaTransformer, CategoricalLabelTransformer, ReplaceValueTransformer, save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        '''
        This is data transformation function
        '''
        try:
            categorical_col = ['propertyType',
                               'locality',
                               'furnishing',
                               'city',
                               'bedrooms',
                               'bathrooms',
                               'RentOrSale',
                               'exactPrice',
                               ]
            # Creating the initial pipeline

            imp_feature = ['propertyType',
                           'locality',
                           'furnishing',
                           'city',
                           'bedrooms',
                           'bathrooms',
                           'RentOrSale',
                           "exactPrice"]

            Initial_pipline = Pipeline(
                steps=[('replace', ReplaceValueTransformer(9, np.nan)),
                       ('replace2', ReplaceValueTransformer("9", np.nan)),
                       ('dropna', DropNaTransformer(subset=["exactPrice"])),
                       #    ('date_transform', DateTransformTransformer(
                       #     date_column='postedOn')),
                       ('fill_na', FillnaTransformer(
                        columns=imp_feature, value="Missing")),
                       ('categorical_label_transform',
                           CategoricalLabelTransformer(categorical_col)),

                       ]
            )

            logging.info(f"Imp features : {imp_feature}")
            logging.info(f"categorical_col : {categorical_col}")

            preprocessor = ColumnTransformer(
                [
                    ("Initial_pipeline", Initial_pipline, imp_feature),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            dataset = pd.read_csv(self.config.dataset_path)

            logging.info(f"Read raw data path {self.config.dataset_path}")
            logging.info("Reading preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            # logging.info(f"dataset columns : {dataset.columns}")
            Input_fea = ['propertyType',
                         'locality',
                         'furnishing',
                         'city',
                         'bedrooms',
                         'bathrooms',
                         'RentOrSale',
                         "exactPrice"]

            data = preprocessing_obj.fit_transform(
                dataset[Input_fea])

            # logging.info(f"Saved preprocessing object. {data}")
            logging.info(f"saving processor : {preprocessing_obj}")

            # Saving the file just to see if processed data is valid for model training
            DF = pd.DataFrame(data, columns=Input_fea)

            # save the dataframe as a csv file
            DF.to_csv("artifacts/processed_data.csv", index=False)

            # Saving preprocessor in artifacts, which is untracked by git
            file_path = self.config.preprocessor_path,
            obj = preprocessing_obj

            save_object(file_path[0], obj)

            # Saving preprocessor in preprocessor directory, which is tracked by git

            file_path = self.config.tracked_preprocessor_path,
            obj = preprocessing_obj

            save_object(file_path[0], obj)

        except Exception as e:
            raise CustomException(e, sys)
