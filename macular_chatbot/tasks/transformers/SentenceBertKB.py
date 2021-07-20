from loguru import logger
from sentence_transformers import SentenceTransformer, util


class SentenceBertKB:
    def __init__(self, input_data, model):
        self.model = SentenceTransformer(model)

        all_facts = [content for id_question, content in input_data.items()]
        all_ids = [id_question for id_question, content in input_data.items()]

        self.embeddings = self.model.encode(all_facts)

        self.kb = dict()

        for id_fact, fact, embedding in zip(all_ids, all_facts, self.embeddings):
            self.kb[len(self.kb)] = {
                "fact": fact,
                "embedding": embedding,
            }

    def encode_sentence(self, input_question):
        question_embedding = self.model.encode(input_question)

        return question_embedding

    def get_closest(self, input_question_embedding):
        hits = util.semantic_search(
            input_question_embedding, self.embeddings, score_function=util.cos_sim
        )

        retrieved_answer = self.kb[hits[0][0]["corpus_id"]]["fact"]

        probability_retrieved_answer = hits[0][0]["score"]

        print(retrieved_answer)

        if probability_retrieved_answer < 0.50:
            retrieved_answer = "I don't know the answer to that one! :("

        return retrieved_answer, probability_retrieved_answer


# class SentenceBertKB:
#     def __init__(self, input_data):
#         self.model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

#         all_questions = [
#             content["question"] for id_question, content in input_data.items()
#         ]
#         all_answers = [content["answer"] for id_question, content in input_data.items()]
#         all_extra = [
#             content["extra_info"] for id_question, content in input_data.items()
#         ]
#         self.embeddings = self.model.encode(all_questions)

#         self.kb = dict()

#         for question, answer, embedding, extra_info in zip(
#             all_questions, all_answers, self.embeddings, all_extra
#         ):
#             self.kb[len(self.kb)] = {
#                 "question": question,
#                 "answer": answer,
#                 "embedding": embedding,
#                 "extra_info": extra_info,
#             }

#     def encode_sentence(self, input_question):
#         question_embedding = self.model.encode(input_question)

#         return question_embedding

#     def get_closest(self, input_question_embedding):
#         hits = util.semantic_search(
#             input_question_embedding, self.embeddings, score_function=util.cos_sim
#         )

#         retrieved_answer = self.kb[hits[0][0]["corpus_id"]]["answer"]

#         probability_retrieved_answer = hits[0][0]["score"]

#         extra_info = self.kb[hits[0][0]["corpus_id"]]["extra_info"]

#         print(probability_retrieved_answer)

#         if probability_retrieved_answer < 0.60:
#             retrieved_answer = "I don't know the answer to that one! :("

#         return retrieved_answer, probability_retrieved_answer, extra_info
