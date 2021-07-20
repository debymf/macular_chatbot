# python -m macular_chatbot.flows.training.train_liveqa_flow --model="msmarco-distilbert-base-v4" --batch_size=16 --epochs=1 --scoring_function --loss

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
from macular_chatbot.flows.evaluation.basic_qa_test_flow import run_test_flow

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

parser.add_argument(
    "--batch_size",
    metavar="Batch size",
    type=int,
    nargs="?",
    default=16,
)


parser.add_argument(
    "--epochs",
    metavar="Number of epochs",
    type=int,
    nargs="?",
    default=5,
)


parser.add_argument(
    "--scoring_function",
    metavar="scoring function",
    type=str,
    nargs="?",
    default="cos",
)

parser.add_argument(
    "--loss",
    metavar="Loss function",
    type=str,
    nargs="?",
    help="Loss function",
    default="ContrastiveLoss",
)


args = parser.parse_args()

LOSS_FUNCTION = args.loss
USED_MODEL = args.model
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

SCORING_FUNCTION = args.scoring_function

if "cos" in SCORING_FUNCTION:
    score_function_eval = util.cos_sim
else:
    score_function_eval = util.dot_score


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
    dataloader = generate_data_loader_task(output_pairs, batch_size=BATCH_SIZE)
    train_sentence_transformers(
        dataloader,
        USED_MODEL,
        MODEL_OUTPUT,
        NUM_EPOCHS,
        LOSS_FUNCTION,
        SCORING_FUNCTION,
    )


FlowRunner(flow=flow1).run()

run_test_flow(used_model=MODEL_OUTPUT, score_function=score_function_eval)
