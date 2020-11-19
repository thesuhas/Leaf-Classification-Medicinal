from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import logging
import sys
from PIL import Image
import cv2
import numpy as np
import io

IMAGE_SIZE = 100

# Loading model
model = tf.keras.models.load_model("")
model.make_predict_function()

app = Flask(__name__, static_folder="C:/Users/suhas/Documents/College Projects/Leaf-Classification-Medicinal/")

def label(prediction):
    prediction = prediction.argmax()
    if prediction == 0:
        return 'pubescent bamboo'
    elif prediction == 1:
        return 'chinese horse chestnut'
    elif prediction == 2:
        return 'anhui barberry'
    elif prediction == 3:
        return 'chinese redbud'
    elif prediction == 4:
        return 'true indigo'
    elif prediction == 5:
        return 'japanese maple'
    elif prediction == 6:
        return 'nanmu'
    elif prediction == 7:
        return 'castor aralia'
    elif prediction == 8:
        return 'chinese cinnamon'
    elif prediction == 9:
        return 'goldenrain tree'
    elif prediction == 10:
        return 'big-fruited holly'
    elif prediction == 11:
        return 'japanese cheesewood'
    elif prediction == 12:
        return 'wintersweet'
    elif prediction == 13:
        return 'camphor tree'
    elif prediction == 14:
        return 'japan arrowwood'
    elif prediction == 15:
        return 'sweet osmanthus'
    elif prediction == 16:
        return 'deodar'
    elif prediction == 17:
        return 'gingko'
    elif prediction == 18:
        return 'crepe myrtle'
    elif prediction == 19:
        return 'oleander'
    elif prediction == 20:
        return 'yew plum pine'
    elif prediction == 21:
        return 'japanese flowering cherry'
    elif prediction == 22:
        return 'glossy privet'
    elif prediction == 23:
        return 'chinese toon'
    elif prediction == 24:
        return 'peach'
    elif prediction == 25:
        return 'ford woodlotus'
    elif prediction == 26:
        return 'trident maple'
    elif prediction == 27:
        return 'beales barberry'
    elif prediction == 28:
        return 'southern magnolia'
    elif prediction == 29:
        return 'canadian poplar'
    elif prediction == 30:
        return 'chinese tulip tree'
    elif prediction == 31:
        return 'tangerine'

def predict(img, model):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    thresh, imgBW = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imgInv = cv2.bitwise_not(imgBW)
    kernel = np.ones((50, 50))
    imgClosed = cv2.morphologyEx(imgInv, cv2.MORPH_CLOSE, kernel)
    # Resize
    new = cv2.resize(imgClosed, (IMAGE_SIZE, IMAGE_SIZE))
    #Adding third dimension to shape
    new.shape = (1,) + new.shape + (1,)
    print(new.shape, flush=True)
    pred = model.predict(new)
    return pred



@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def pred_image():
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img))
    # Converting image to np array
    img = np.array(img)
    print(img.shape, flush=True)
    species = predict(img, model)
    species = label(species)
    print(species, flush=True)
    response = {"Prediction": species}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)