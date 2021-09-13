from loguru import logger
from sentence_transformers import SentenceTransformer, util


class SentenceBertKB:
    def __init__(self, input_data, short_answers, model):
        self.model = SentenceTransformer(model)

        all_facts = [content for id_question, content in input_data.items()]
        all_ids = [id_question for id_question, content in input_data.items()]
        all_short_answers = [content["short"] for id_, content in short_answers.items()]
        all_long_answers = [content["long"] for id_, content in short_answers.items()]

        self.embeddings = self.model.encode(all_facts)

        self.kb = dict()

        for id_fact, fact, short_answer, long_answer, embedding in zip(all_ids, all_facts, all_short_answers, all_long_answers,self.embeddings):
            self.kb[len(self.kb)] = {
                "fact": fact,
                "short": short_answer,
                "long": long_answer,
                "embedding": embedding,
            }

    def encode_sentence(self, input_question):
        TOKENIZERS_PARALLELISM=False
        question_embedding = self.model.encode(input_question)

        return question_embedding

    def get_closest(self, input_question_embedding):
        hits = util.semantic_search(
            input_question_embedding, self.embeddings, score_function=util.cos_sim
        )

        retrieved_answer = self.kb[hits[0][0]["corpus_id"]]["fact"]
        retrieved_short = self.kb[hits[0][0]["corpus_id"]]["short"]
        retrieved_long = self.kb[hits[0][0]["corpus_id"]]["long"]

        if retrieved_long=='':
            retrieved_long = False

        probability_retrieved_answer = hits[0][0]["score"]

        if probability_retrieved_answer < 0.50:
            retrieved_answer = "I don't know the answer to that one! :("

        return retrieved_answer, probability_retrieved_answer, retrieved_short, retrieved_long


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
