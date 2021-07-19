from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util, losses, evaluation
from .SentenceBertKB import SentenceBertKB
from tqdm import tqdm


class TrainSentenceTransformerTask(Task):
    def run(self, data_for_training, model_name, output_location):
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            data_for_training["sentences1"],
            data_for_training["sentences2"],
            data_for_training["scores"],
        )

        model = SentenceTransformer(model_name)
        train_loss = losses.CosineSimilarityLoss(model)

        # Tune the model
        model.fit(
            train_objectives=[(data_for_training["dataloader"], train_loss)],
            epochs=1,
            warmup_steps=10,
            output_path=output_location,
            save_best_model=True,
            evaluator=evaluator,
            evaluation_steps=500,
        )
