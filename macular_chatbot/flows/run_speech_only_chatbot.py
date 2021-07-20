# python -m macular_chatbot.flows.run_speech_only_chatbot

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import PrepareDataTask
from macular_chatbot.tasks.transformers import (
    ConvertToVectorsTask,
    EvaluateVectorsTask,
    GenerateKBTask,
)
from sentence_transformers import util
from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings
import argparse

checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "chatbot_flow"
file_location = settings["basic_qa_data"]
USED_MODEL = "./models/msmarco-distilbert-base-tas-b_live_qa"


prepare_data_task = PrepareDataTask()
gen_kb = GenerateKBTask()
evaluate_vectors = EvaluateVectorsTask()
with Flow("Testing model without fine-tuning") as flow1:
    input_data = prepare_data_task(file_location)
    output_vectors = gen_kb(input_data["facts"])
    evaluate_vectors(output_vectors, score_function=score_function)

FlowRunner(flow=flow1).run()
