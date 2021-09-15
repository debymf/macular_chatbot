# AI for Visually Impaired AIVI (EVE) Dialogue System

The repo contains an end-to-end dialogue system designed for giving an easy way for visually impaired people to get information about their condition. The focus is on Macular Degeneration.

## Dataset

The dataset is contained in the ``data` folder, it was scraped from several FAQS on official websites (such as the NHS) on topics related to Macular Degeneration. However, the code here can be easily adapted to work with other datasets, as long as it follows the same structure. I also included the LiveQA dataset and MedQUAD dataset for more generic health related topics.

## Installing the requirements

```
pip install -r requirements.txt
```

## Training the retrieval model

In order to train the model that will retrieve the correct answers based on the user's query, the following command should be run:

```
python -m macular_chatbot.flows.training.similar_answers.train_macular_flow --model="sentence-transformers/all-mpnet-base-v2" --batch_size=16 --epochs=5 --scoring_function="cos" --loss="ContrastiveLoss"

```

You can try with different models and parameters, but the ones shown above got the best results with my dataset.

To train the model using LIVE QA or MedQUAD, the following commands can be used:

```
python -m macular_chatbot.flows.training.similar_answers.train_medquad_flow --model="sentence-transformers/all-mpnet-base-v2" --batch_size=16 --epochs=5 --scoring_function="cos" --loss="ContrastiveLoss"

```

After the model is trained, it will print values for Recall to assess its performance.

## Calibrate audio sensitivity

In order to make sure the audio is being captured properly, we need to calibrate the audio, by setting up the variable `energy_threshold`
The `recognizer_instance.energy_threshold` is basically how sensitive the recognizer is to when recognition should start. While it is possible to automatically calibrate the audio using the `adjust_for_ambient_noise`, I found better results calibrating it myself.

Run the following code, and stop it when you are happy with the recognizer performance (for example, when it stops lagging between your voice and the recognition):

```
python macular_chatbot/scripts/calibrate_audio.py
```

## Running the dialogue system

After the model is ready, you can run the dialogue system but running:

```
python -m macular_chatbot.flows.run_speech_only_chatbot --model="./models/all-mpnet-base-v2_macular" --audio_energy=600
```

Where `model` should be replace with the model location, and `audio_energy` is the value obtained from the calibration.
