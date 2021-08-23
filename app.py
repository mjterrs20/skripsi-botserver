import json
import rake
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

# load data using Python JSON module
with open('chatbot_eksyar.json','r') as f:
    data = json.loads(f.read())

# load cnn model
model = load_model("cnn_model.h5")
labels = ['asuransi', 'bank', 'eksyar', 'investasi', 'reksadana']

# Tokonizer
datas = []
for i in range(len(data['items'])):
    datas.append(data['items'][i]["questions"])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(datas)

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
        return jsonify({
            'label' : label,
        })
    except KeyError as e:
        return str(e)

@app.route("/rake", methods= ['POST'])
def test_rake():
    keywords = r.run(request.form['quest'])
    return str(keywords)

def cnn_predict(quest):
    puretext = tokenizer.texts_to_sequences([quest])
    text_pad = pad_sequences(puretext,maxlen=50,padding='post')
    predicted = model.predict(text_pad)
    index_label = predicted.argmax()
    return labels[index_label]