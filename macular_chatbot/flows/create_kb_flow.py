# python -m macular_chatbot.flows.create_kb_flow

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import PrepareDataTask
from macular_chatbot.tasks.transformers import ConvertToVectorsTask
from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings

checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "chatbot_flow"

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

prepare_data_task = PrepareDataTask()
convert_to_vectors = ConvertToVectorsTask()

with Flow("Running example flow") as flow1:
    parsed_data = prepare_data_task()
    final_vectors = convert_to_vectors()


FlowRunner(flow=flow1).run()
