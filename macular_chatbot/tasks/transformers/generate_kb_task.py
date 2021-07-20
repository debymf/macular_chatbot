from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from .SentenceBertKB import SentenceBertKB


class GenerateKBTask(Task):
    def run(self, input_data, model_name):
        logger.info("** Converting Sentences to Vector **")
        kb = SentenceBertKB(input_data, model_name)

        return kb
