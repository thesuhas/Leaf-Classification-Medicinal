# Leaf-Classification-Medicinal

## About

Built an SVM and a CNN that classifies leaves. Also built a website using Flask that does webscraping with beautifulsoup to obtain the medicinal properties of the plant after classification.

## Pre-Processing and Feature Extraction

Pre-processed the Leaf images by converting to Grayscale, applied a Gaussian Blur, then Binary, Filled Holes, Inverted the Image. <br>
A contour was drawn to extract Perimeter, Area, Aspect Ratio, among other features. <br>
Colour-based features such as Mean and Variance of Red, Green and Blue channels independently. <br>
Width and Height and dependent features were extracted on drawing a Bounding Box.<br>
Texture-based features such as Contrast, Correlation, Entropy and idf were extracted. <br>

## SVM

A SVM was built and optimised using a GridSearch and the best parameters were found were C=10, Radial Basis Function Kernel, and Gamma=0.1. <br>
Obtained a Test Score of 98.88% and Training Score of 97.48%.

## CNN

A CNN was built with 3 Convolutional Layers each with a MaxPool Layer and a Batch Normalisation Layer. Adams Optimiser was used along with He Normal Initialisation. A dense layer was added at the end with a Batch Normalisation Layer before the Output (SoftMax) layer.<br> Obtained a Training Score of 97.4% along with a Validation Score of 80%.

## Website

A simple Flask-based Website was built where the User uploads a picture of a Leaf. The CNN would then classify the image and give the class which would be used to webscrape the Latin Names and its Medicinal Properties (if any).