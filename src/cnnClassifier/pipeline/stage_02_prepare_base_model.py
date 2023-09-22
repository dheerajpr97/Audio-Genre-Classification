from cnnClassifier.config.config import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)


if __name__ == "__main__":
    try:
        logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
    except Exception as e:
        logger.exception(e)
        logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
        raise e