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


class SpeechClassifierTask(Task):
    def run(self):
        logger.info("*** Running speech classifier ***")
        nltk.download("popular", quiet=True)
        nltk.download("nps_chat", quiet=True)
        nltk.download("punkt")
        nltk.download("wordnet")

        posts = nltk.corpus.nps_chat.xml_posts()[:10000]

        # To Recognise input type as QUES.
        def dialogue_act_features(post):
            features = {}
            for word in nltk.word_tokenize(post):
                features["contains({})".format(word.lower())] = True
            return features

        featuresets = [
            (dialogue_act_features(post.text), post.get("class")) for post in posts
        ]
        size = int(len(featuresets) * 0.1)
        train_set, test_set = featuresets[size:], featuresets[:size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)

        return classifier
