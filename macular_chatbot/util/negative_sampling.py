from loguru import logger
import random
from .bm25 import BM25Fit, BM25Search


def gen_negative_pairs(questions, all_answers, num_neg=100):

    fit_class = BM25Fit()
    ix = fit_class.run(all_answers)
    search_class = BM25Search()
    retrieval_results = search_class.run(all_answers, ix, limit=num_neg)

    return retrieval_results