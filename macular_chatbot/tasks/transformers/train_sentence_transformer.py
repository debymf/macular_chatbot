from prefect import Task
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    util,
    losses,
    evaluation,
)
from .SentenceBertKB import SentenceBertKB
from tqdm import tqdm


class TrainSentenceTransformerTask(Task):
    def run(
        self,
        data_for_training,
        model_name,
        output_location,
        num_epochs=5,
        loss_function="ContrastiveLoss",
        scoring_function="cos",
    ):
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            data_for_training["sentences1"],
            data_for_training["sentences2"],
            data_for_training["scores"],
        )

        model = SentenceTransformer(model_name)

        if "CosineSimilarityLoss" in loss_function:
            train_loss = losses.CosineSimilarityLoss(model)
        else:
            if "cos" in scoring_function:
                distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
            else:
                distance_metric = losses.SiameseDistanceMetric.EUCLIDEAN

            train_loss = losses.ContrastiveLoss(model, distance_metric=distance_metric)

        # Tune the model
        model.fit(
            train_objectives=[(data_for_training["dataloader"], train_loss)],
            epochs=num_epochs,
            output_path=output_location,
            save_best_model=True,
            evaluator=evaluator,
        )
