from HousePricePredictRecommend.config.configuration import ConfigurationManager
from HousePricePredictRecommend.components.model_trainer import ModelTrainer
from HousePricePredictRecommend import logger


STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTrainer(config=model_training_config)
        model_training.initiate_model_trainer()
        model_training.initiate_model_trainer_rent()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
