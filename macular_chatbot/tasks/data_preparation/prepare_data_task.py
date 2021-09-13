from prefect import Task
from loguru import logger
import pandas as pd
import random


class PrepareDataTask(Task):
    def run(self, input_file_location):
        logger.info("** Running Prepare Data Task **")
        input_data = pd.read_csv(input_file_location)

        facts = list()
        logger.info("Creating KB with facts.")
        for index, row in input_data.iterrows():
            facts.append(row["answer"])

        facts = list(set(facts))
        random.shuffle(facts)

        facts_mapping = dict()
        facts_kb = dict()

        count = 0
        for f in facts:
            facts_mapping[f] = count
            facts_kb[count] = f
            count += 1

        questions = dict()
        qa_mapping = dict()
        short_answer = dict()

        for index, row in input_data.iterrows():
            questions[index] = row["question"]
            qa_mapping[index] = facts_mapping[row["answer"]]
            short_answer[facts_mapping[row["answer"]]] = {"short": row["answer_short"], "long": row["answer_long"]}

        return {"questions": questions, "facts": facts_kb, "mapping": qa_mapping, "short_answer": short_answer}


# class PrepareDataTask(Task):
#     def run(self, input_file_location):
#         logger.info("** Running Prepare Data Task **")
#         input_data = pd.read_csv(input_file_location)
#         output_dict = dict()

#         for index, row in input_data.iterrows():
#             output_dict[index] = {
#                 "question": row["question"],
#                 "answer": row["Answer"],
#                 # "extra_info": row["Extra"],
#             }
#         return output_dict
