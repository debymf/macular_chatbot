from prefect import Task
from loguru import logger
import pandas as pd


class PrepareDataTask(Task):
    def run(self, input_file_location):
        logger.info("** Running Prepare Data Task **")
        input_data = pd.read_csv(input_file_location)
        output_dict = dict()

        for index, row in input_data.iterrows():
            output_dict[index] = {
                "question": row["Question"],
                "answer": row["Answer"],
                "extra_info": row["Extra"],
            }
        return output_dict
