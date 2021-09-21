### python macular_chatbot/scripts/calibrate_audio.py

import speech_recognition as sr
from loguru import logger


def get_sentence_input(r):
    user_response = None
    with sr.Microphone() as source:
        logger.info("Say anything.")
        audio = r.listen(source, phrase_time_limit=10)
        logger.info("Got Audio!")
    try:
        user_response = format(r.recognize_google(audio))
        print("\033[91m {}\033[00m".format("YOU SAID : " + user_response))
    except sr.UnknownValueError:
        logger.error("Audio not recognized")
        pass
    except Exception as e:
        print(e)


r = sr.Recognizer()
logger.info("Starting calibration of audio")
input("Press any key to start.")
r.dynamic_energy_threshold = False
for value in range(50, 1000, 50):
    logger.debug(f"Trying energy_threshold = {value}")
    r.energy_threshold = value

    get_sentence_input(r)
    reply = input("Continue calibrating? (y/n)")
    if reply != "y":
        logger.info(f"Finishing callirationg! Value for audio_energy: {value}")
        break
