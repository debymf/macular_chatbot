from prefect import Task
from loguru import logger
import pandas as pd
import random
import xml.etree.ElementTree as ET
from macular_chatbot.util import gen_negative_pairs
from tqdm import tqdm
import glob, os


class GenNegativePairsTask(Task):
    def run(self, positive_pairs):
        logger.info("** Generating negative pairs **")

        # gen the negative pairs
        all_answers = dict()
        all_questions = dict()
        count = 0

        logger.info("** Generating negative pairs **")

        all_answers_list = list()
        for p in tqdm(positive_pairs):
            all_questions[count] = p[0]
            all_answers_list.append(p[1])
            count = count + 1

        count = 0
        for a in tqdm(all_answers_list):
            all_answers[count] = a
            count = count + 1

        retrieved_results = gen_negative_pairs(
            questions=all_questions, all_answers=all_answers
        )

        count = 0
        negative_pairs = []
        for i in range(0, len(all_questions)):
            closest = list(retrieved_results[i].keys())[1:]
            for n in closest:
                negative_pairs.append([all_questions[i], all_answers[n]])

        logger.info(f"Total negative: {len(negative_pairs)}")
        logger.info(f"Total positive: {len(positive_pairs)}")

        return {"negative": negative_pairs, "positive": positive_pairs}
