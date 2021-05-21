from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from telegram import Update, ForceReply
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)


class TelegramBotTask(Task):
    def run(self, kb):
        self.kb = kb
        logger.info("*** Starting Telegram Bot ***")
        """Start the bot."""

        # Create the Updater and pass it your bot's token.
        updater = Updater("1652848597:AAGjM1wQidYvTjXsgrst5VMVfTj2Q5uFi68")

        # Get the dispatcher to register handlers
        dispatcher = updater.dispatcher

        # on different commands - answer in Telegram
        dispatcher.add_handler(CommandHandler("start", self.start))
        dispatcher.add_handler(CommandHandler("help", self.help_command))

        # on non command i.e message - echo the message on Telegram

        dispatcher.add_handler(
            MessageHandler(Filters.text & ~Filters.command, self.echo)
        )

        # Start the Bot
        updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        updater.idle()

    # Define a few command handlers. These usually take the two arguments update and
    # context.
    @staticmethod
    def start(update: Update, _: CallbackContext) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        update.message.reply_markdown_v2(
            fr"Hi {user.mention_markdown_v2()}\!",
            reply_markup=ForceReply(selective=True),
        )

    @staticmethod
    def help_command(update: Update, _: CallbackContext) -> None:
        """Send a message when the command /help is issued."""
        update.message.reply_text("Help!")

    def echo(self, update: Update, _: CallbackContext) -> None:
        """Echo the user message."""
        question_embedding = self.kb.encode_sentence(update.message.text)
        predicted_answer = self.kb.get_closest(question_embedding)

        update.message.reply_text(predicted_answer)
