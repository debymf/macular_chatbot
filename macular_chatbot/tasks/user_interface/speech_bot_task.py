from prefect import Task
from loguru import logger
from sentence_transformers import SentenceTransformer, util
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os

warnings.filterwarnings("ignore")
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer


# colour palet
def prRed(skk):
    print("\033[91m {}\033[00m".format(skk))


def prGreen(skk):
    print("\033[92m {}\033[00m".format(skk))


def prYellow(skk):
    print("\033[93m {}\033[00m".format(skk))


def prLightPurple(skk):
    print("\033[94m {}\033[00m".format(skk))


def prPurple(skk):
    print("\033[95m {}\033[00m".format(skk))


def prCyan(skk):
    print("\033[96m {}\033[00m".format(skk))


def prLightGray(skk):
    print("\033[97m {}\033[00m".format(skk))


def prBlack(skk):
    print("\033[98m {}\033[00m".format(skk))


class SpeechBotTask(Task):

    # Keyword Matching

    @staticmethod
    def greeting(sentence):
        GREETING_INPUTS = (
            "hello",
            "hi",
            "greetings",
            "sup",
            "what's up",
            "hey",
        )
        GREETING_RESPONSES = [
            "hi",
            "hey",
            "*nods*",
            "hi there",
            "hello",
            "I am glad! You are talking to me",
        ]
        """If user's input is a greeting, return a greeting response"""
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)

    @staticmethod
    def get_sentence_input(r):
        user_response = None
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            user_response = format(r.recognize_google(audio))
            print("\033[91m {}\033[00m".format("YOU SAID : " + user_response))
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print(e)
        return user_response

    @staticmethod
    def read_sentence(predicted_answer):
        file = "reply.wav"
        tts = gTTS(text=predicted_answer, lang="en", tld="com")
        tts.save(file)
        os.system("mpg123 " + file)

    @staticmethod
    def dialogue_act_features(post):
        features = {}
        for word in nltk.word_tokenize(post):
            features["contains({})".format(word.lower())] = True
        return features

    def run(self, kb, classifier):
        self.kb = kb
        logger.info("*** Starting speech based bot ***")
        r = sr.Recognizer()
        flag = True
        start_intro = "My name is Jarvis. I will answer your queries about Macular degeneration. If you want to exit, say Bye"

        self.read_sentence(start_intro)

        print("\033[93m {}\033[00m".format("Jarvis: " + start_intro))

        while flag == True:
            user_response = self.get_sentence_input(r)
            if user_response:
                clas = classifier.classify(self.dialogue_act_features(user_response))
                if clas != "Bye":
                    if clas == "Emotion":
                        flag = False
                        prYellow("Jarvis: You are welcome..")
                    else:
                        if self.greeting(user_response) != None:

                            greeting_response = self.greeting(user_response)
                            self.read_sentence(greeting_response)
                            print(
                                "\033[93m {}\033[00m".format(
                                    "Jarvis: " + greeting_response
                                )
                            )
                        else:
                            input_embedding = self.kb.encode_sentence(user_response)
                            (
                                predicted_answer,
                                score,
                            ) = self.kb.get_closest(input_embedding)

                            self.read_sentence(predicted_answer)
                            print(
                                "\033[93m {}\033[00m".format(
                                    "Jarvis said:" + predicted_answer
                                )
                            )
                else:
                    flag = False
                    prYellow("Jarvis: Bye! take care..")
                    self.read_sentence("Bye! take care..")

    # def echo(self, update: Update, _: CallbackContext) -> None:

    #     if score > 0.6 and self.extra_info != "None":
    #         keyboard = [
    #             [
    #                 InlineKeyboardButton("Yes", callback_data="1"),
    #                 InlineKeyboardButton("No", callback_data="2"),
    #             ]
    #         ]

    #         reply_markup = InlineKeyboardMarkup(keyboard)

    #         update.message.reply_text(
    #             "Do you want more details?", reply_markup=reply_markup
    #         )