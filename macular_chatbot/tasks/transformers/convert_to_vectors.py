from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from .SentenceBertKB import SentenceBertKB


class ConvertToVectorsTask(Task):
    def run(self, input_data, model):
        logger.info("** Converting Sentences to Vector **")
        model = SentenceTransformer(model)

        logger.info("** Embedding Questions **")
        embedding_questions = model.encode(
            input_data["questions"], show_progress_bar=True
        )
        logger.info("** Embedding Facts **")
        embedding_facts = model.encode(input_data["facts"], show_progress_bar=True)

        print(len(input_data["facts"]))

        return {
            "questions": input_data["questions"],
            "questions_embedding": embedding_questions,
            "facts": input_data["facts"],
            "facts_embedding": embedding_facts,
            "mapping": input_data["mapping"],
        }
