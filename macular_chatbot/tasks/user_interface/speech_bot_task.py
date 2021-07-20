from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util


class SpeechBotTask(Task):
    @staticmethod
    def get_sentence_input():
        return "What is dry AMD?"

    def run(self, kb):
        self.kb = kb
        logger.info("*** Starting speech based bot ***")
        input = self.get_sentence_input()
        input_embedding = self.kb.encode_sentence(input)
        (
            predicted_answer,
            score,
        ) = self.kb.get_closest(input_embedding)

        print(input)
        print(predicted_answer)

    def echo(self, update: Update, _: CallbackContext) -> None:

        if score > 0.6 and self.extra_info != "None":
            keyboard = [
                [
                    InlineKeyboardButton("Yes", callback_data="1"),
                    InlineKeyboardButton("No", callback_data="2"),
                ]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            update.message.reply_text(
                "Do you want more details?", reply_markup=reply_markup
            )
