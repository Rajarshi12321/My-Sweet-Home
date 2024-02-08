from HousePricePredictRecommend.config.configuration import ConfigurationManager
from HousePricePredictRecommend.components.data_transformation_recommend_data import DataTransformationRecommend
from HousePricePredictRecommend import logger


STAGE_NAME = "Data Transformation for recommendation data stage"


class DataTransformationRecommendPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_recommend_config()
        data_transformation = DataTransformationRecommend(
            config=data_transformation_config)
        data_transformation.initiate_data_transformation_recommend()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationRecommendPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
