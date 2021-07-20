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
<<<<<<< HEAD
from macular_chatbot.tasks.user_interface import SpeechBotTask, SpeechClassifierTask
=======
from macular_chatbot.tasks.user_interface import SpeechBotTask
>>>>>>> 29004202fc5f430d9602bb66512600b0db526c7c

checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "speech_chatbot_flow"
file_location = settings["basic_qa_data"]
USED_MODEL = "./models/msmarco-distilbert-base-v4_live_qa"


<<<<<<< HEAD
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)


prepare_data_task = PrepareDataTask(**cache_args)
gen_kb = GenerateKBTask(**cache_args)
chatbot_task = SpeechBotTask()
speech_classifier = SpeechClassifierTask(**cache_args)
with Flow("Testing speech based chatbot") as flow1:
    input_data = prepare_data_task(file_location)
    kb = gen_kb(input_data["facts"], USED_MODEL)
    s_classifier = speech_classifier()
    chatbot_task(kb, s_classifier)
=======
prepare_data_task = PrepareDataTask()
gen_kb = GenerateKBTask()
chatbot_task = SpeechBotTask()
with Flow("Testing speech based chatbot") as flow1:
    input_data = prepare_data_task(file_location)
    kb = gen_kb(input_data["facts"], USED_MODEL)
    chatbot_task(kb)
>>>>>>> 29004202fc5f430d9602bb66512600b0db526c7c

FlowRunner(flow=flow1).run()
