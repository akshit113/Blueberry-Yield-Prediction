import pickle
from flask import Flask, request,app, jsonify,url_for,render_template
import numpy


app=Flask(__name__)

loaded_model = pickle.load(open('linear-regression-model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)

