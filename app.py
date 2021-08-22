from rake_nltk import Rake
from keras.models import load_model
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)



# load cnn model
model = load_model("cnn_model.h5")



# install Rake
r = Rake(language="indonesian", min_length=1, max_length=5)

# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')



@app.route("/rake", methods= ['POST'])
def login():
    try: 
        r.extract_keywords_from_text(request.form['quest'])
        keyword = r.get_ranked_phrases()
        print(keyword)
        return str(keyword)
    except KeyError as e:
        return str(e)


