from cnnClassifier.config.config import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation"   

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.get_test_data()
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == "__main__":
    try:
        logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
        pipeline = EvaluationPipeline()
        pipeline.main()
        logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
    except Exception as e:
        logger.exception(e)
        logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
        raise e
