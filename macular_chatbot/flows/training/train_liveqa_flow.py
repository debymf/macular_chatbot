# python -m macular_chatbot.flows.training.train_liveqa_flow

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import PrepareDataLiveQATask
from macular_chatbot.tasks.transformers import (
    PreparePairsForTrainingTask,
    TrainSentenceTransformerTask,
)

from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings

checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "train_liveqa_flow"
file_location = settings["live_qa_train"]
USED_MODEL = "msmarco-distilbert-base-v4"
MODEL_OUTPUT = "./models/" + USED_MODEL + "_live_qa"
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

prepare_data_task = PrepareDataLiveQATask()
generate_data_loader_task = PreparePairsForTrainingTask()
train_sentence_transformers = TrainSentenceTransformerTask()

with Flow("Training model with LiveQA") as flow1:
    output_pairs = prepare_data_task(file_location)
    dataloader = generate_data_loader_task(output_pairs)
    train_sentence_transformers(dataloader, USED_MODEL, MODEL_OUTPUT)
    # evaluate_vectors(output_vectors)


FlowRunner(flow=flow1).run()
