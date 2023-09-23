from cnnClassifier.config.config import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.train_model import Training     
from cnnClassifier import logger

STAGE_NAME = "Train Model"

class TrainModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_model()
        training.get_datasets()
        #training.train_valid_generator()
        training.train(
            callback_list=callback_list
    )


if __name__ == "__main__":
    try:
        logger.info(f">>> Stage: {STAGE_NAME} started! <<<")
        pipeline = TrainModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>> Stage: {STAGE_NAME} completed! <<<")
    except Exception as e:
        logger.exception(e)
        logger.error(f"Exception occurred in {STAGE_NAME} stage: {e}")
        raise e
