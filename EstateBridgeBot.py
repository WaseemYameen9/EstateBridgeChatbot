import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('training_data/intents.json').read())

words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/chatbot_model.h5')

class MessageInput(BaseModel):
    message: str

def clean_up_sentence(sentence):
    sentence_words = nltk.wordpunct_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array([bag])

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]).reshape(1, -1))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
       return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results] if results else [{'intent': 'unknown', 'probability': '1.0'}]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't know about that."

# print("GO! Bot is running!")

# while True:
#     try:
#         message = input("You: ")  # Added prompt
#         if message.lower() == "exit":
#             print("Bot: Goodbye!")
#             break  # Exit condition

#         print(f"User input received: {message}")  # Debugging step

#         ints = predict_class(message)
#         print(f"Predicted class: {ints}")  # Debugging step

#         if ints:  # Ensure we have a prediction before calling get_response
#             res = get_response(ints, intents)
#         else:
#             res = "I'm sorry, I didn't understand that."

#         print("Bot:", res)
#     except Exception as e:
#         print(f"Error: {e}")
@app.post("/chat/")
def chat_with_bot(user_message: MessageInput):
    ints = predict_class(user_message.message)
    response = get_response(ints, intents)
    return {"response": response}
    