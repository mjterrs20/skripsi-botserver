from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/login", methods= ['POST'])
def login():
    try: 
        return jsonify({
            'nama' : request.form['username'],
            'passu' : request.form['pass'],
        })
    except KeyError as e:
        return str(e)

