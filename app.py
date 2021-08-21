import numpy as np
import json
import pandas as pd
import math
from rake_nltk import Rake
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
from collections import Counter


app = Flask(__name__)

# load data using Python JSON module
with open('chatbot_eksyar.json','r') as f:
    data = json.loads(f.read())
df = pd.json_normalize(data, record_path =['items'])

# load cnn model
model = load_model("cnn_model.h5")

# Prepation Tokenizer
question = df.questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(question)
sequences = tokenizer.texts_to_sequences(question)
word_index = tokenizer.word_index

# label
le = LabelEncoder()
label = df.labels
labelEncode = le.fit_transform(label)

# install Rake
r = Rake(language="indonesian", min_length=1, max_length=5)

# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/chat", methods= ['POST'])
def login():
    try: 
        label = cnn_predict(request.form['quest'])
        npQuestion , npAnswer = get_df(label[0])
        tempRake, qRake = rake_question(npQuestion, request.form['quest'])
        tempCounter, counterQuestUser = counter_result(tempRake, qRake)
        maxScore, indexQuest = score_cosine(tempCounter, counterQuestUser)
        return jsonify({
            'label' : label[0],
            'max_score ': str(maxScore),
            'quest': npQuestion[indexQuest],
            'ans': npAnswer[indexQuest],
        })
    except KeyError as e:
        return str(e)


def cnn_predict(quest):
    puretext = tokenizer.texts_to_sequences([quest])
    text_pad = pad_sequences(puretext,maxlen=50,padding='post')
    predicted = model.predict(text_pad)
    predicted_category = predicted.argmax(axis=1)
    return le.classes_[predicted_category]

def counter(quest):
    counter = Counter(quest)
    return counter

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)
def get_df(labels):
    dfNews = df.loc[df['labels'] == labels]
    npQuestions = dfNews.questions.to_numpy()
    npAnswers = dfNews.answers.to_numpy()
    return npQuestions,npAnswers

def rake_question(npQuestions, questUser):
    tempRake = []
    for i in range(len(npQuestions)):
        r.extract_keywords_from_text(npQuestions[i])
        tempRake.append(r.get_ranked_phrases())
    r.extract_keywords_from_text(questUser)
    qRake = r.get_ranked_phrases()
    return tempRake, qRake
    
def counter_result(tempRake, qRake):
    tempCounter = []
    for i in range(len(tempRake)):
        tempCounter.append(counter(tempRake[i]))
    counterQuestUser = counter(qRake)
    return tempCounter, counterQuestUser

def score_cosine(tempCounter, counterQuestUser):
    scoresCosine = []
    for i in range(len(tempCounter)):
        scoresCosine.append(counter_cosine_similarity(counterQuestUser, tempCounter[i]) * 100)
    npScore = np.array(scoresCosine)
    maxScore = np.amax(npScore)
    indexQuest = np.argmax(npScore)
    return maxScore, indexQuest