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
        all_answers = dict()
        count = 0
        logger.info("** Preparing MedQuAD dataset **")
        for filename in glob.iglob(f"{input_file_location}/**", recursive=True):
            if os.path.isfile(filename):  # filter dirs
                tree = ET.parse(filename)
                root = tree.getroot()
                for content in root.findall("QAPairs"):

                    if content.find("QAPair"):
                        question = content.find("QAPair").find("Question")

                    if content.find("QAPair"):
                        answer = content.find("QAPair").find("Answer")

                    if question.text and answer.text:
                        positive_pairs.append([question.text, answer.text])
                        all_answers[count] = answer.text
                        count = count + 1

        logger.info(f"Total positive: {len(positive_pairs)}")

        return positive_pairs
