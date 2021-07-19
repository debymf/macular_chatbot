# python -m macular_chatbot.flows.training.train_medquad_flow --model="msmarco-distilbert-base-v4"

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import PrepareDataMedQuadTask
from macular_chatbot.tasks.transformers import (
    PreparePairsForTrainingTask,
    TrainSentenceTransformerTask,
)
import argparse
from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings

checkpoint_dir = settings["checkpoint_dir"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    metavar="Model name",
    type=str,
    nargs="?",
    help="sentence embedding model",
    default="msmarco-distilbert-base-v4",
)
args = parser.parse_args()

USED_MODEL = args.model

TASK_NAME = "train_medquad_flow"
file_location = settings["medquad_train"]
MODEL_OUTPUT = "./models/" + USED_MODEL + "_medquad"
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

prepare_data_task = PrepareDataMedQuadTask()
generate_data_loader_task = PreparePairsForTrainingTask()
train_sentence_transformers = TrainSentenceTransformerTask()

with Flow("Training model with MedQuAD") as flow1:
    output_pairs = prepare_data_task(file_location)
    dataloader = generate_data_loader_task(output_pairs)
    train_sentence_transformers(dataloader, USED_MODEL, MODEL_OUTPUT)


FlowRunner(flow=flow1).run()
