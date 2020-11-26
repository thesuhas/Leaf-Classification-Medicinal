from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import logging
import sys
from PIL import Image
import cv2
import numpy as np
import io
import requests
from bs4 import BeautifulSoup
import re

IMAGE_SIZE = 100

# Loading model
model = tf.keras.models.load_model("C:/Users/suhas/Documents/College Projects/Leaf-Classification-Medicinal/model")
model.make_predict_function()

app = Flask(__name__, static_folder="C:/Users/suhas/Documents/College Projects/Leaf-Classification-Medicinal/")

def label(prediction):
    prediction = prediction.argmax()
    if prediction == 0:
        return ('pubescent bamboo', 'Phyllostachys edulis')
    elif prediction == 1:
        return ('chinese horse chestnut', 'aesculus chinensis')
    elif prediction == 2:
        return ('anhui barberry', 'berberis anhweiensis ahrendt')
    elif prediction == 3:
        return ('chinese redbud', 'cercis chinensis')
    elif prediction == 4:
        return ('true indigo', 'indigofera tinctoria')
    elif prediction == 5:
        return ('japanese maple', 'acer Palmatum')
    elif prediction == 6:
        return ('nanmu', 'phoebe nanmu')
    elif prediction == 7:
        return ('castor aralia', 'Kalopanax septemlobus')
    elif prediction == 8:
        return ('chinese cinnamon', 'cinnamomum japonicum')
    elif prediction == 9:
        return ('goldenrain tree', 'koelreuteria paniculata')
    elif prediction == 10:
        return ('big-fruited holly', 'ilex macrocarpa')
    elif prediction == 11:
        return ('japanese cheesewood', 'pittosporum tobira')
    elif prediction == 12:
        return ('wintersweet', 'chimonanthus praecox')
    elif prediction == 13:
        return ('camphor tree', 'cinnamomum camphora')
    elif prediction == 14:
        return ('japan arrowwood', 'viburnum awabuki')
    elif prediction == 15:
        return ('sweet osmanthus', 'osmanthus fragrans')
    elif prediction == 16:
        return ('deodar', 'cedrus deodara')
    elif prediction == 17:
        return ('gingko', 'ginkgo biloba')
    elif prediction == 18:
        return ('crepe myrtle', 'lagerstroemia indica')
    elif prediction == 19:
        return ('oleander', 'nerium oleander')
    elif prediction == 20:
        return ('yew plum pine', 'podocarpus macrophyllus')
    elif prediction == 21:
        return ('japanese flowering cherry', 'prunus serrulata')
    elif prediction == 22:
        return ('glossy privet', 'ligustrum lucidum')
    elif prediction == 23:
        return ('chinese toon', 'tonna sinensis')
    elif prediction == 24:
        return ('peach', 'prunus persica')
    elif prediction == 25:
        return ('ford woodlotus', 'manglietia fordiana')
    elif prediction == 26:
        return ('trident maple', 'acer buergerianum')
    elif prediction == 27:
        return ('beales barberry', 'mahonia bealei')
    elif prediction == 28:
        return ('southern magnolia', 'magnolia grandiflora')
    elif prediction == 29:
        return ('canadian poplar', 'populus ×canadensis')
    elif prediction == 30:
        return ('chinese tulip tree', 'liriodendron chinense')
    elif prediction == 31:
        return ('tangerine', 'citrus reticulata')

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
    new.shape = (1,) + new.shape + (1, )
    #print(new.shape, flush=True)
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
    img2 = np.array(img)
    species = predict(img2, model)
    (species, sci_name) = label(species)
    
    # Web Scraping
    sci_name= sci_name.replace(' ','+')
    url= 'https://pfaf.org/user/Plant.aspx?LatinName={}'.format(sci_name)
    page= requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    results= soup.find(class_='boots2').find('span')

    resarr= list(results.descendants)
    sci_name = sci_name.split('+')
    sci_name = sci_name[0] + ' ' + sci_name[1]
    try: 
        medprop= resarr[-1]
        medprop= re.sub('\[.*?\]',' ',medprop)
        return render_template("pred.html", medprop=medprop, species=species, sci_name=sci_name)
    except IndexError: 
        return render_template("pred.html", medprop="No Medicinal Properties Available", species=species, sci_name=sci_name)

if __name__ == '__main__':
    app.run(debug=True)