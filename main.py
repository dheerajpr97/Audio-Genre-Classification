from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

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