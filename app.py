import json
import rake
import math
import random
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


app = Flask(__name__)

# load data using Python JSON module
dataset = []
questions = [] # menggunkan question yang telah di streaming sebelumnya
with open('dataset.json','r') as f:
    data = json.loads(f.read())

# add dataset from data
for i in range(len(data['items'])):
    dataset.append(data['items'])

# load cnn model
model = load_model("model.h5")
labels = ['armuzna', 'badal', 'dam', 'haji', 'ihram', 'jumrah','miqat', 'perempuan', 'sai', 'tahallul', 'tempat_khusus', 'thawaf', 'umrah']

# Keterangan
keterangan = ['Maaf marbot tidak dapat menjawab, silahkan tanya lagi tentang',
             'Marbot tidak dapat menjawab, sialahkan tanyakan mengenai',
             'Marbot tidak mengerti silahkan bertanya tentang',
             'Marbot belum paham dengan yang anda tanyakan silahkan tenyakan lagi tentang',
             'Marbot saat ini hanya bisa menjawab beberapa hal saja, silahkan tanyakan lagi mengenai']

# Tokonizer
for i in range(len(dataset)):
    questions.append(data['items'][i]["normalized"])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

# Load data normalization
# Data yang beberapa akan dilakukan normalisasi
with open('normalisasi.json','r') as f:
    data_normalization = json.loads(f.read())

# Inisialisasi Untuk Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
    
# Load Rake
r = rake.Rake("Stopword.txt")

# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/test", methods= ['POST'])
def test():
    result_ww = preprocesing(request.form['quest'])
    return result_ww

@app.route("/chat", methods= ['POST'])
def question():
    try: 
        preQuest = preprocesing(request.form['quest'])
        label = cnn_predict(preQuest)
        npQuestion , npAnswer = get_df(label)
        tempRake, qRake = rake_question(npQuestion, preQuest)
        tempCounter, counterQuestUser = counter_result(tempRake, qRake)
        maxScore, indexQuest = score_cosine(tempCounter, counterQuestUser)

        # cek apakah maxScore sama dengan 0 atau tidak
        if maxScore != 0:
            return jsonify({
                'label' : label,
                'max_score ': str(maxScore),
                'question_rake': qRake,
                'quest': npQuestion[indexQuest],
                'ans': npAnswer[indexQuest],
            })
        else:
           return jsonify({
                'label' : label,
                'max_score ': str(maxScore),
                'question_rake': qRake,
                'quest': npQuestion[indexQuest],
                'ans': keterangan[random.randint(0,4)] + " "+ label,
            }) 
    except KeyError as e:
        return str(e)

@app.route("/rake", methods= ['POST'])
def test_rake():
    keywords = r.run(request.form['quest'])
    return str(keywords)

# ========   START TEXT PREPOCESSING   =============
# Melakukan Preprocesing terlebih dahulu terhadap Question
# 1. Regex (hapus tanda baca & lowercase)
# 2. Normalization

character = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~\'0123456789'
# Defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in character])
    return punctuationfree

# Fungsi untuk melakukan Normalisasi
def normalization(tokens):
    for idx, items in enumerate(tokens):
        for j in range(len(data_normalization['items'])):
            if items == data_normalization['items'][j]['before']:
                tokens[idx] = data_normalization['items'][j]['after']
    result_normalized = " ".join(tokens)
    return result_normalized

# Tahapan Preprocesing
def preprocesing(quest):
    # Regex & Case Folding
    res = remove_punctuation(quest).lower()
    # Tokenizing
    tokens = res.split()
    # Normalisasi 
    norm = normalization(tokens)
    # Steamming (Menggunakan Library Sastrawi)
    # steamming = stemmer.stem(norm)
    return norm

# ========  END TEXT PREPOCESSING   =============


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
            npQuestions.append(data['items'][i]['normalized'])
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

# 4. Konver menjadi counter
def counter(quest):
    counter = Counter(quest)
    return counter

# 5. Mencari cosine similiarity
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
