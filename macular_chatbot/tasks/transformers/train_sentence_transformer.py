from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util, losses
from .SentenceBertKB import SentenceBertKB
from tqdm import tqdm


class TrainSentenceTransformerTask(Task):
    def run(self, dataloader, model_name, output_location):

        model = SentenceTransformer(model_name)
        train_loss = losses.CosineSimilarityLoss(model)

        # Tune the model
        model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=5,
            warmup_steps=100,
            output_path=output_location,
            save_best_model=True,
        )
