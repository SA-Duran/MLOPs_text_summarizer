from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging.logger import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"Entrada a main inicializa configuration")
        config = ConfigurationManager()
        logger.info(f"llamada de información a config")
        model_trainer_config = config.get_model_trainer_config()
        logger.info(f"llamada a model trainer")
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        logger.info(f"llamada a train del modelo")
        model_trainer_config.train()
        logger.info(f"termina train")