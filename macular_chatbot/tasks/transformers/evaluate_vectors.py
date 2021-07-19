from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from .SentenceBertKB import SentenceBertKB
from tqdm import tqdm


class EvaluateVectorsTask(Task):
    @staticmethod
    def get_closest(question_embedding, facts_embedding):
        retrieved = util.semantic_search(
            question_embedding, facts_embedding, score_function=util.cos_sim, top_k=5
        )

        return retrieved

    @staticmethod
    def get_recall_at_k(retrieved, answer, k):
        retrieved = retrieved[:k]
        if answer in retrieved:
            return 1
        else:
            return 0

    def run(self, input_data):
        total_questions = 0
        recall_at_1 = 0
        recall_at_3 = 0
        recall_at_5 = 0
        for content in tqdm(input_data["questions_embedding"]):
            retrieved = self.get_closest(content, input_data["facts_embedding"])[0]
            retrieved_index = [result["corpus_id"] for result in retrieved]
            answer = input_data["mapping"][total_questions]

            recall_at_1 += self.get_recall_at_k(retrieved_index, answer, 1)
            recall_at_3 += self.get_recall_at_k(retrieved_index, answer, 3)
            recall_at_5 += self.get_recall_at_k(retrieved_index, answer, 5)

            # print("================")
            # print("Question")
            # print(input_data["questions"][total_questions])
            # print("=========")
            # print("Answer")
            # print(input_data["facts"][answer])
            # print(answer)
            # print("=========")
            # print("Retrieved")
            # print(retrieved)
            # print("Recall at 1")
            # print(recall_at_1)
            # print("Recall at 3")
            # print(recall_at_3)
            # print("Recall at 5")
            # print(recall_at_5)
            # input()
            total_questions += 1

        logger.info("*** Recall at 1 ***")
        logger.info(recall_at_1 / total_questions)
        logger.info("*** Recall at 3 ***")
        logger.info(recall_at_3 / total_questions)
        logger.info("*** Recall at 5 ***")
        logger.info(recall_at_5 / total_questions)
