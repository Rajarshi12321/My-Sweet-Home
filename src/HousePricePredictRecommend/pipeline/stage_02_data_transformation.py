from HousePricePredictRecommend.config.configuration import ConfigurationManager
from HousePricePredictRecommend.components.data_transformation import DataTransformation
from HousePricePredictRecommend import logger


STAGE_NAME = "Data Preprocessing/Transformation Stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)
        data_transformation.initiate_data_transformation()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
