from HousePricePredictRecommend.constants import *
from HousePricePredictRecommend.utils.common import read_yaml, create_directories
from HousePricePredictRecommend.entity.config_entity import DataIngestionConfig, DataTransformationConfig, TrainingConfig, DataTransformationRecommendConfig


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            dataset_path=config.dataset_path,
            preprocessor_path=config.preprocessor_path,
            tracked_preprocessor_path=config.tracked_preprocessor_path
        )

        return data_transformation_config

    def get_model_training_config(self) -> TrainingConfig:
        config = self.config.training

        create_directories([config.root_dir])

        model_training_config = TrainingConfig(
            root_dir=config.root_dir,
            training_data=config.training_data,
            trained_model_file_path=config.trained_model_file_path,
            trained_model_file_path_rent=config.trained_model_file_path_rent,
            mlflow_uri=config.mlflow_uri,
            model_params=self.params

        )

        return model_training_config

    def get_data_transformation_recommend_config(self) -> DataTransformationRecommendConfig:
        config = self.config.recommed_data_preprocessing

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationRecommendConfig(
            root_dir=config.root_dir,
            dataset_path=config.dataset_path,
            processed_dataset_path=config.processed_dataset_path,
            recommend_dataset_path=config.recommend_dataset_path,
            tracked_recommend_dataset_path=config.tracked_recommend_dataset_path
        )

        return data_transformation_config
