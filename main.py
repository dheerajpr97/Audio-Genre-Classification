from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_data import PrepareDataTrainingPipeline
from cnnClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
except Exception as e:
    logger.exception(e)
    logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Prepare Data Stage"
try:
    logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
    pipeline = PrepareDataTrainingPipeline()
    pipeline.main()
    logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
except Exception as e:
    logger.exception(e)
    logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
    raise e

STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.main()
    logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
except Exception as e:
    logger.exception(e)
    logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
    raise e

