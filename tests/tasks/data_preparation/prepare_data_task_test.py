import unittest
from macular_chatbot.tasks.data_preparation import PrepareDataTask
from macular_chatbot.tasks.transformers import ConvertToVectorsTask
from macular_chatbot.tasks.user_interface import TelegramBotTask
from dynaconf import settings


class PrepareDataTaskTest(unittest.TestCase):
    def test_prep_data_task(self):
        prep = PrepareDataTask()
        parsed_data = prep.run(input_file_location=settings["input_data"])

        convert_data = ConvertToVectorsTask()
        kb = convert_data.run(input_data=parsed_data)

        telegram_bot_task = TelegramBotTask()

        telegram_bot_task.run(kb=kb)

        print(kb[0])
