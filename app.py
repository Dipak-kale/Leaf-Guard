from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from keras.preprocessing import image
import keras.utils as image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer







MODEL_PATH="plant_model.h5"
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224,224))
    xtest_image = image.img_to_array(xtest_image)
    test_image = np.expand_dims(xtest_image, axis=0)
    a = np.argmax(model.predict(test_image ),axis=1)
    return   a



from flask import Flask, request,render_template
app = Flask(__name__, template_folder='template')  # still relative to module
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    ans =["Apple scab","Aplle Black rot","Cedar apple rust","Apple Healthy","Blueberry healthy","Cherry Powdery mildew","Cherry healthy","Corn Cercospora leaf spot Gray leaf spot","Corn Common rust","Corn Northern Leaf Blight","Corn healthy","Grape  Black rot","Grape  Esc (Black Measles)","Grape  Leaf blight (Isariopsis Leaf Spot)","Grape  healthy","Orange  Haunglongbing ","Peach  Bacterial spot","Peach  healthy","Pepper, bell  Bacterial spot","Pepper, bell  healthy","Potato  Early blight","Potato  Late blight","Potato  healthy","Raspberry  healthy","Soybean  healthy","Squash  Powdery mildew","Strawberry  Leaf scorch","Strawberry  healthy","Tomato  Bacterial spot","Tomato  Early blight","Tomato  Late blight","Tomato  Leaf Mold","Tomato  Septoria leaf spot","Tomato  Spider mites Two-spotted spider mite","Tomato  Target Spot","Tomato  Tomato Yellow Leaf Curl Virus","Tomato  Tomato mosaic virus","Tomato  healthy"]
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])
       
       
        return ans[preds[0]]
    return None
@app.route('/detection_model', methods=['GET','POST'])
def detection_model():
    # Main page
    return render_template('detection_model.html')
@app.route('/DiseasesANDCure', methods=['GET','POST'])
def DiseasesANDCure():
    # Main page
    return render_template('DiseasesANDCure.html')

if __name__ == '__main__':
    app.run(debug=True)