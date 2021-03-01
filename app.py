from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('predict_blood1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    #print(int_features)
    #print(final)
    predictions = model.predict(final)
    #print(predictions)
    output = predictions[0]
    #output = "{:f}".format(predictions[0][1])
    #print(output)
    return render_template('predict.html', prediction_text="Prediction is {}".format(output))

    if predictions == 0:
        return render_template('predict.html', prediction_text=f'The donor will not donate blood in near future')
    else:
        return render_template('predict.html', prediction_text=f'The donor will donate blood in near future')


if __name__ == '__main__':
    app.run(debug=True)
