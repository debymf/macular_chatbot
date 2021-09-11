from prefect import Task
from loguru import logger
import pandas as pd
import random
import xml.etree.ElementTree as ET
from macular_chatbot.util import gen_negative_pairs
from tqdm import tqdm
import glob, os


class PrepareDataMacularTask(Task):
    def run(self, input_file_location):
        logger.info("** Running Prepare Data Task for Macular dataset **")
        input_data = pd.read_csv(input_file_location)

        positive_pairs = list()
        for index, row in input_data.iterrows():
            positive_pairs.append([row["question"], row["answer"]])

        return positive_pairs
