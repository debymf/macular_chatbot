# python -m macular_chatbot.flows.training.train_liveqa_flow --model="msmarco-distilbert-base-v4"

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import (
    PrepareDataLiveQATask,
    GenNegativePairsTask,
)
from macular_chatbot.tasks.transformers import (
    PreparePairsForTrainingTask,
    TrainSentenceTransformerTask,
)
import argparse
from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings


checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "train_liveqa_flow"
file_location = settings["live_qa_train"]

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

MODEL_OUTPUT = "./models/" + USED_MODEL + "_live_qa"
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

prepare_data_task = PrepareDataLiveQATask()
generate_data_loader_task = PreparePairsForTrainingTask()
gen_negative_pairs = GenNegativePairsTask()
train_sentence_transformers = TrainSentenceTransformerTask()

with Flow("Training model with LiveQA") as flow1:
    positive_pairs = prepare_data_task(file_location)
    output_pairs = gen_negative_pairs(positive_pairs)
    dataloader = generate_data_loader_task(output_pairs)
    train_sentence_transformers(dataloader, USED_MODEL, MODEL_OUTPUT)


FlowRunner(flow=flow1).run()
