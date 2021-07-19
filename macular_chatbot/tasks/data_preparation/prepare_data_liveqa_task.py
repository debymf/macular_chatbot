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
        all_answers = list()

        # read the files
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
            all_answers.append(answer)

        # gen the negative pairs

        negative_pairs = []
        logger.info("** Generating negative pairs **")
        for p in tqdm(positive_pairs):
            negative_pairs.extend(
                gen_negative_pairs(
                    question=p[0], all_answers=all_answers, positive_answer=p[1]
                )
            )

        logger.info(f"Total negative: {len(negative_pairs)}")
        logger.info(f"Total positive: {len(positive_pairs)}")

        return {"negative": negative_pairs, "positive": positive_pairs}
