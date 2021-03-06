from flask import Flask, render_template, request, jsonify, send_from_directory
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
from werkzeug.utils import secure_filename

IMAGE_SIZE = 100

# Loading model
model = tf.keras.models.load_model("C:\\Users\\suhas\\Documents\\College Projects\\Leaf-Classification-Medicinal\\model")
model.make_predict_function()

UPLOAD_FOLDER = 'C:\\Users\\suhas\\Documents\\College Projects\\Leaf-Classification-Medicinal\\upload\\'

app = Flask(__name__, static_folder="C:\\Users\\suhas\\Documents\\College Projects\\Leaf-Classification-Medicinal\\")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def label(prediction):
    prediction = prediction.argmax()
    print(prediction)
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
    elif prediction == 32:
        return ('European field elm', 'ulmus carpinifolia')
    elif prediction == 33:
        return ('Woolly flowered maple', 'acer erianthum')
    elif prediction == 34:
        return ('Round-eared Willow', 'salix aurita')
    elif prediction == 35:
        return ('White oak', 'quercus alba')
    elif prediction == 36:
        return ('Grey Alder', 'alnus incana')
    elif prediction == 37:
        return ('Downy Birch', 'betula pubescens')
    elif prediction == 38:
        return ('White Willow', 'salix alba')
    elif prediction == 39:
        return ('Aspen', 'populus tremula')
    elif prediction == 40:
        return ('Wych or Scotch Elm', 'ulmus glabra')
    elif prediction == 41:
        return ('Common Rowan', 'sorbus aucuparia')
    elif prediction == 42:
        return ('Gray willow', 'salix cinerea')
    elif prediction == 43:
        return ('Eastern Cottonwood', 'populus deltoides')
    elif prediction == 44:
        return ('Small-leaved Lime', 'tilia cordata')
    elif prediction == 45:
        return ('Swedish Whitebeam', 'sorbus intermedia')
    elif prediction == 46:
        return ('European Beech', 'fagus sylvatica')
  

    

def predict(img, model):
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    #thresh, imgBW = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #imgInv = cv2.bitwise_not(imgBW)
    #kernel = np.ones((50, 50))
    #imgClosed = cv2.morphologyEx(imgInv, cv2.MORPH_CLOSE, kernel)
    # Resize
    new = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    #Adding third dimension to shape
    #new.shape = (1,) + new.shape + (1, )
    new.shape = (1, ) + new.shape
    #print(new.shape, img.shape,flush=True)
    pred = model.predict(new)
    #print(pred, flush= True)
    return pred

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def pred_image():
    file_img = request.files['file']
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img))
    # Converting image to np array
    img2 = np.array(img)
    img2 = np.divide(img2, 255)
    print(img2, flush=True)
    species = predict(img2, model)
    #print("Prediction: ", species, flush=True)
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
    img = img.resize((300, 300))
    print(file_img.filename[:-3] + "jpg", flush=True)
    filename = file_img.filename[:-3] + "jpg"
    img.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    #print(os.path.join(app.config['UPLOAD_FOLDER'], file_img.filename), flush=True)
    #print(file_img.filename, flush=True)
    try: 
        medprop= resarr[-1]
        medprop= re.sub('\[.*?\]',' ',medprop)
        return render_template("pred.html", medprop=medprop, species=species, sci_name=sci_name, img=filename)
    except IndexError: 
        return render_template("pred.html", medprop="No Medicinal Properties Available", species=species, sci_name=sci_name, img=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
