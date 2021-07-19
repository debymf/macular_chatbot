from loguru import logger
import random


def gen_negative_pairs(question, all_answers, positive_answer, num_neg=50):
    no_duplicate = 1
    negative_pairs = []
    while no_duplicate == 1:
        negative_selected = random.sample(all_answers, num_neg)
        if positive_answer not in negative_selected:
            no_duplicate = 0
            for n in negative_selected:
                negative_pairs.append([question, n])

    return negative_pairs