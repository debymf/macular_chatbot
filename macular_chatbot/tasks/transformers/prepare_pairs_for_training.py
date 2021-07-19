from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util, InputExample
from torch.utils.data import DataLoader
from .SentenceBertKB import SentenceBertKB


class PreparePairsForTrainingTask(Task):
    def run(self, input_data):
        logger.info("** Converting Pairs for training with Sentence Transformers**")
        train_examples = list()
        for negative in input_data["negative"]:
            train_examples.append(
                InputExample(texts=[negative[0], negative[1]], label=0.0)
            )
        for positive in input_data["positive"]:
            train_examples.append(
                InputExample(texts=[positive[0], positive[1]], label=1.0)
            )

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        return train_dataloader
