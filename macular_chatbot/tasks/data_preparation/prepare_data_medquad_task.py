from prefect import Task
from loguru import logger
import pandas as pd
import random
import xml.etree.ElementTree as ET
from macular_chatbot.util import gen_negative_pairs
from tqdm import tqdm
import glob, os


class PrepareDataMedQuadTask(Task):
    def run(self, input_file_location):
        positive_pairs = list()
        all_answers = list()
        logger.info("** Preparing MedQuAD dataset **")
        for filename in glob.iglob(f"{input_file_location}/**", recursive=True):
            if os.path.isfile(filename):  # filter dirs
                tree = ET.parse(filename)
                root = tree.getroot()
                for content in root.findall("QAPairs"):

                    question = content.find("QAPair")

                    answer = content.find("QAPair")

                    if question and answer:
                        positive_pairs.append(
                            [question.find("Question").text, answer.find("Answer").text]
                        )
                        all_answers.append(answer.text)

        # gen the negative pairs
        negative_pairs = []
        logger.info("** Generating negative pairs **")
        for p in tqdm(positive_pairs):
            negative_pairs.append(
                gen_negative_pairs(
                    question=p[0], all_answers=all_answers, positive_answer=p[1]
                )
            )

        logger.info(f"Total negative: {len(negative_pairs)}")
        logger.info(f"Total positive: {len(positive_pairs)}")

        return {"negative": negative_pairs, "positive": positive_pairs}
