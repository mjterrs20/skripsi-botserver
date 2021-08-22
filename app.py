from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# load cnn model
model = load_model("cnn_model.h5")

# Tokonizer
corpus = ["Bagaimana ketentuan istishna?", "Apa yang dimaksud ijarah?", "Bagaimana cara kerja ijarah?"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')



@app.route("/rake", methods= ['POST'])
def login():
    try: 
        res = cnn_predict(request.form['quest'])
        return res 
    except KeyError as e:
        return str(e)

def cnn_predict(quest):
    puretext = tokenizer.texts_to_sequences([quest])
    text_pad = pad_sequences(puretext,maxlen=50,padding='post')
    predicted = model.predict(text_pad)
    return str(predicted)