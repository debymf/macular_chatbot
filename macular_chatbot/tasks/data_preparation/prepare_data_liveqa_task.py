from prefect import Task
from loguru import logger
import pandas as pd
import random
import xml.etree.ElementTree as ET
from macular_chatbot.util import gen_negative_pairs
from tqdm import tqdm


class PrepareDataLiveQATask(Task):
    def run(self, input_file_location):
        logger.info("** Preparing LiveQA dataset **")
        tree = ET.parse(input_file_location)
        root = tree.getroot()

        positive_pairs = list()
        all_answers = dict()

        # read the files

        count = 0
        for content in root.findall("NLM-QUESTION"):
            question = content.find("MESSAGE").text
            if not question:
                question = content.find("SUBJECT").text
            answer = (
                content.find("SUB-QUESTIONS")
                .find("SUB-QUESTION")
                .find("ANSWERS")
                .find("ANSWER")
                .text
            )
            positive_pairs.append([question, answer])
            all_answers[count] = answer
            count += 1

        return positive_pairs
