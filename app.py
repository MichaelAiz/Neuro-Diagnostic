from werkzeug.utils import secure_filename
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from flask import Flask, request, flash, redirect, render_template
from code.preprocessing import process_single_img

app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def scan_alzheimers():
    if 'file' not in request.files:
        app.logger.debug('not found file')
        return redirect(request.url)
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('E:/Projects/Neuro-Diagnostic/', filename))
    img =  process_single_img(os.path.join('E:/Projects/Neuro-Diagnostic/', filename))
    model = keras.models.load_model('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/FineTunedModelNoAugment.h5')
    prediction = model.predict(img)
    return str(prediction)



if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/scanCancer', methods = ['GET', 'POST'])
# def scan_cancer()
