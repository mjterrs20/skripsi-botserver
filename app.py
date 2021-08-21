from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/join")
def join():
    return "<p>bambang pamungkas</p>"

@app.route("/login", methods= ['POST'])
def login():
    try: 
        return jsonify({
            'nama' : request.form['username'],
            'passu' : request.form['pass'],
        })
    except KeyError as e:
        return str(e)