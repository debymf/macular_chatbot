# python -m macular_chatbot.flows.evaluation.basic_qa_test_flow --model="./models/msmarco-distilbert-base-v4_live_qa" --scoring="dot"

from prefect import Flow
import prefect
from loguru import logger
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from macular_chatbot.tasks.data_preparation import PrepareDataTask
from macular_chatbot.tasks.transformers import ConvertToVectorsTask, EvaluateVectorsTask
from sentence_transformers import util
from prefect import Flow, Parameter, Task, tags, task
from dynaconf import settings
import argparse

checkpoint_dir = settings["checkpoint_dir"]
TASK_NAME = "chatbot_flow"
file_location = settings["basic_qa_data"]
# USED_MODEL = "paraphrase-mpnet-base-v2"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    metavar="Model name",
    type=str,
    nargs="?",
    help="sentence embedding model",
    default="msmarco-distilbert-base-v4",
)

parser.add_argument(
    "--scoring",
    metavar="Scoring function",
    type=str,
    nargs="?",
    help="Scoring function",
    default="cos",
)


args = parser.parse_args()

scoring_function = args.scoring

if "cos" in scoring_function:
    score_function = util.cos_sim
else:
    score_function = util.dot_score


USED_MODEL = args.model

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)

prepare_data_task = PrepareDataTask()
convert_to_vectors = ConvertToVectorsTask()
evaluate_vectors = EvaluateVectorsTask()


def run_test_flow(used_model, score_function):
    with Flow("Testing model without fine-tuning") as flow1:
        input_data = prepare_data_task(file_location)
        output_vectors = convert_to_vectors(input_data, USED_MODEL)
        evaluate_vectors(output_vectors, score_function=score_function)

    FlowRunner(flow=flow1).run()


if __name__ == "__main__":
    run_test_flow(used_model=USED_MODEL, score_function=score_function)