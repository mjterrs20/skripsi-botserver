from flask import Flask, request, jsonify, render_template


app = Flask(__name__)



# route duniawi
@app.route("/")
def hello_world():
    return render_template('index.html')



@app.route("/rake", methods= ['POST'])
def login():
    try: 
        return request.form['quest']
    except KeyError as e:
        return str(e)


