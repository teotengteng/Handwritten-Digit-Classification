import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
#import math
import re

from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import load_img
from skimage.transform import resize 

print("Loading model") 
global model 
model = load_model('handwriting.h5') 

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    number = re.search(r"\d",filename)
    actual = number.group()
    #my_img = plt.imread(os.path.join('uploads', filename))
    my_file = os.path.join('uploads', filename)
    img = load_img(my_file, color_mode="grayscale", target_size=(28,28))
    img = np.invert(img)
    img = img.astype('float32')
    img_re = img.reshape(28,28,1)
    img_re /= 255
    img_re = resize(img_re, (28, 28, 1))
    model.run_eagerly=True
    probabilities = model.predict(np.array( [img_re,] ))[0,:]
    print(probabilities)
    number_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index = np.argsort(probabilities)
    #marks = math.trunc(probabilities[index[9]] * 100)
    if number_to_class[index[9]] == actual:
     grade = "Good Job!"
    else:
     grade = "Hmmm ... Did I make a wrong guess?"
    predictions = {
      "actual":actual,
      "digit":number_to_class[index[9]],
      "prob" :probabilities[index[9]],
      "comment":grade
     }
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
