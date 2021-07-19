from loguru import logger
import random
from .bm25 import BM25Fit, BM25Search


def gen_negative_pairs(question, all_answers, positive_answer, num_neg=10):

    fit_class = BM25Fit()
    ix = fit_class.run(all_answers)
    search_class = BM25Search()
    retrieval_results = search_class.run({"id": positive_answer}, ix, limit=num_neg)
    closest = list(retrieval_results["id"].keys())[1:]

    # negative_pairs = []
    # while no_duplicate == 1:
    #     negative_selected = random.sample(all_answers, num_neg)
    #     if positive_answer not in negative_selected:
    #         no_duplicate = 0
    #         for n in negative_selected:
    #             negative_pairs.append([question, n])

    # return negative_pairs

    negative_pairs = []
    for n in closest:
        negative_pairs.append([question, all_answers[n]])

    return negative_pairs