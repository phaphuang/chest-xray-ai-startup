from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import time
import flask
from hashlib import sha1
from werkzeug.utils import secure_filename

from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.models import load_model
import tensorflow as tf

DATA_DIR = 'static/uploads'

app = flask.Flask(__name__, static_url_path="/static", static_folder="static")

#disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
                  # 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
                  # 'Hernia']

model = load_model("model/chest-xray-pneumonia.h5")
global graph
graph = tf.get_default_graph()

def get_rez(loaded_model, pic):
    img = image.load_img(pic, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #print(x)
    with graph.as_default():
        p_good, p_ill = np.around(loaded_model.predict(x), decimals=2)[0]
    return p_good, p_ill


@app.route('/')
def index():
    return render_template('index.html', title='Home')


@app.route('/upload/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        #sha1sum = sha1(flask.request.data).hexdigest()
        #print(sha1sum)
        f.save(os.path.join(DATA_DIR, (secure_filename(f.filename))))
        path = os.path.join(DATA_DIR, f.filename)
        #print(path)
        prob_good, prob_ill = get_rez(model, path)
    return render_template('result.html', img_src=f.filename, prob_good=prob_good*100, prob_ill=prob_ill*100)

if __name__ == '__main__':
    app.debug = True
    app.run('0.0.0.0', threaded=True)