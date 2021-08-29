from werkzeug.utils import secure_filename
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from flask_cors import CORS
from flask import Flask, request, flash, redirect, render_template
from PIL import Image


sys.path.append('E:/Projects/Neuro-Diagnostic/Code/')
from Code import TFrecords
from Code import preprocessing

app = Flask(__name__)
CORS(app)



@app.route('/scan', methods=['POST'])
def scan_alzheimers():
    if 'file' not in request.files:
        app.logger.debug('not found file')
        return redirect(request.url)
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('E:/Projects/Neuro-Diagnostic/', filename))
    img = preprocessing.process_single_img(os.path.join('E:/Projects/Neuro-Diagnostic/', filename))
    model = keras.models.load_model('E:/Projects/Neuro-Diagnostic/Models/FineTunedModelNoAugment.h5')
    prediction = model.predict(img)
    return str(np.argmax(prediction))

if __name__ == '__main__':
    app.run(debug=True)

