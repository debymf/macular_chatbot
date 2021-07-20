from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util, InputExample
from torch.utils.data import DataLoader
from .SentenceBertKB import SentenceBertKB


class PreparePairsForTrainingTask(Task):
    def run(self, input_data, batch_size=16):
        logger.info("** Converting Pairs for training with Sentence Transformers**")
        train_examples = list()
        scores = list()
        sentences1 = list()
        sentences2 = list()
        for negative in input_data["negative"]:
            train_examples.append(
                InputExample(texts=[negative[0], negative[1]], label=0.0)
            )
            sentences1.append(negative[0])
            sentences2.append(negative[1])
            scores.append(0.0)
        for positive in input_data["positive"]:
            train_examples.append(
                InputExample(texts=[positive[0], positive[1]], label=1.0)
            )
            sentences1.append(positive[0])
            sentences2.append(positive[1])
            scores.append(1.0)

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

        return {
            "dataloader": train_dataloader,
            "sentences1": sentences1,
            "sentences2": sentences2,
            "scores": scores,
        }
