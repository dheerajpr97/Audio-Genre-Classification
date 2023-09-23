from cnnClassifier.config.config import ConfigurationManager
from cnnClassifier.components.prepare_data import PrepareData
from cnnClassifier import logger

STAGE_NAME = "Prepare Data Stage"

class PrepareDataTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_data_config = config.get_prepare_data_config()
        prepare_data = PrepareData(config=prepare_data_config)
        music_df = prepare_data.dataframe_create()
        mfcc_df = prepare_data.mfcc_feature_extraction(music_df)
        prepare_data.save_data(mfcc_df)

if __name__ == "__main__":
    try:
        logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
        pipeline = PrepareDataTrainingPipeline()
        pipeline.main()
        logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
    except Exception as e:
        logger.exception(e)
        logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
        raise e
