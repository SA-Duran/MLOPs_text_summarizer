from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.logging.logger import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"Entrada a main")
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        logger.info(f"configurations manager ok")
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        logger.info(f"inicialización de modelo ok entra a evaluar")
        model_evaluation_config.evaluate()
        logger.info(f"termina evaluar")