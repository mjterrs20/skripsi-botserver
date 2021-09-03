import json
import rake
import math
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter


app = Flask(__name__)

# load data using Python JSON module
dataset = []
questions = []
with open('haji_dataset_final_banget.json','r') as f:
    data = json.loads(f.read())

# add dataset from data
for i in range(len(data['items'])):
    dataset.append(data['items'])

# load cnn model
model = load_model("cnn_model_haji_v1.h5")
labels = ['armuzna', 'badal', 'dam', 'haji', 'ihram', 'jumrah','manasik','miqat', 'perempuan', 'sai','sakit','tahalul', 'tempat_khusus', 'thawaf', 'umrah']

# Tokonizer
for i in range(len(dataset)):
    questions.append(data['items'][i]["questions"])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

# Load Rake
r = rake.Rake("Stopword.txt")

# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/chat", methods= ['POST'])
def question():
    try: 
        label = cnn_predict(request.form['quest'])
        npQuestion , npAnswer = get_df(label)
        tempRake, qRake = rake_question(npQuestion, request.form['quest'])
        tempCounter, counterQuestUser = counter_result(tempRake, qRake)
        maxScore, indexQuest = score_cosine(tempCounter, counterQuestUser)
        return jsonify({
            'label' : label,
            'max_score ': str(maxScore),
            'quest': npQuestion[indexQuest],
            'ans': npAnswer[indexQuest],
        })
    except KeyError as e:
        return str(e)

@app.route("/rake", methods= ['POST'])
def test_rake():
    keywords = r.run(request.form['quest'])
    return str(keywords)

# memprediksi label menggunakan Algoritma CNN
def cnn_predict(quest):
    puretext = tokenizer.texts_to_sequences([quest])
    text_pad = pad_sequences(puretext,maxlen=50,padding='post')
    predicted = model.predict(text_pad)
    index_label = predicted.argmax()
    return labels[index_label]

# 1. mendefinisikan terlebih dahulu data
def get_df(label):
    npQuestions = []
    npAnswers = []
    for i in range(len(dataset)):
        if data['items'][i]["labels"] == label:
            npQuestions.append(data['items'][i]['questions'])
            npAnswers.append(data['items'][i]['answers'])
    return npQuestions,npAnswers

# 2. Mengekstraksi npQuest dan pertanyaan user menjadi keyword
# Menerapkan algoritma RAKE
def rake_question(npQuestions, questUser):
    tempRake = []
    for i in range(len(npQuestions)):
        tempRake.append(r.run(npQuestions[i]))
    qRake = r.run(questUser)
    return tempRake, qRake

# 3. Menghitung counter dari list keyword    
def counter_result(tempRake, qRake):
    tempCounter = []
    for i in range(len(tempRake)):
        tempCounter.append(counter(tempRake[i]))
    counterQuestUser = counter(qRake)
    return tempCounter, counterQuestUser

# 4. konver menjadi counter
def counter(quest):
    counter = Counter(quest)
    return counter

# 5. mencari cosine similiarity
def score_cosine(tempCounter, counterQuestUser):
    scoresCosine = []
    for i in range(len(tempCounter)):
        scoresCosine.append(counter_cosine_similarity(counterQuestUser, tempCounter[i]) * 100)
    npScore = np.array(scoresCosine)
    maxScore = np.amax(npScore)
    indexQuest = np.argmax(npScore)
    return maxScore, indexQuest

# 6. Algoritma Cosine Similiarity
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)



    


