# program to capture single image from webcam in python 

# importing OpenCV library 
import datetime
import os
from cv2 import *
from cv2 import VideoCapture
from cv2 import imshow
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyWindow
import cv2
from flask import Flask, render_template, request, url_for
from keras.applications import VGG16
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16

def vgg_std16_model(img_rows, img_cols, color_type=3):
    nb_classes = 10
    # Remove fully connected layer and replace
    # with softmax for classifying 10 classes
    vgg16_model = VGG16(weights="imagenet", include_top=False)

    # Freeze all layers of the pre-trained model
    for layer in vgg16_model.layers:
        layer.trainable = False

    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation = 'softmax')(x)

    model = Model(vgg16_model.input, predictions)

    return model
def get_cv2_image(path, img_rows, img_cols, color_type=3):
    # Loading as Grayscale image
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Reduce size
    img = cv2.resize(img, (img_rows, img_cols))
    return img
def plot_vgg16_test_class(model, test_files, image_number):
    img_brute = test_files[image_number]
    batch_size=40
	
    nb_epoch=10
    img_rows = 64
    img_cols = 64
    color_type = 1
    activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}
	#batch_size = 40
    #nb_epoch = 10

    im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (img_rows,img_cols)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)

    img_display = cv2.resize(img_brute,(img_rows,img_cols))
    plt.imshow(img_display, cmap='gray')

    y_preds = model.predict(im, batch_size=batch_size, verbose=1)
    print(y_preds)
    y_prediction = np.argmax(y_preds)
    print('Y Prediction: {}'.format(y_prediction))
    print('Predicted as: {}'.format(activity_map.get('c{}'.format(y_prediction))))

    plt.show()
# initialize the camera 
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.static_folder = 'static'

@app.route('/home', methods=['GET', 'POST'])
def diet():
     return render_template('home.html')

from werkzeug.utils import secure_filename
@app.route('/capture', methods=['GET', 'POST'])
def capture():
     

        f = request.files['ifile']       
        filename = secure_filename(f.filename)
        rimg=filename
        f.save("static/" + rimg)
        batch_size=40
        nb_epoch=10
        img_rows = 64
        img_cols = 64
        model_vgg16 = vgg_std16_model(img_rows, img_cols)
        model_vgg16.load_weights('weights_best_vgg16.hdf5')
        color_type = 1
        dt=datetime.datetime.now()
        arr=str(dt).split(" ")
        dt=arr[0]
        activity_map = {'c0': 'Safe driving',
                    'c1': 'Texting - right',
                    'c2': 'Talking on the phone - right',
                    'c3': 'Texting - left',
                    'c4': 'Talking on the phone - left',
                    'c5': 'Operating the radio',
                    'c6': 'Drinking',
                    'c7': 'Reaching behind',
                    'c8': 'Hair and makeup',
                    'c9': 'Talking to passenger'}
        #batch_size = 40
        #nb_epoch = 10 
        img_brute =get_cv2_image("static/"+rimg, img_rows, img_cols, color_type)
        im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (img_rows,img_cols)).astype(np.float32) / 255.0
        im = np.expand_dims(im, axis =0)

        img_display = cv2.resize(img_brute,(img_rows,img_cols))
        plt.imshow(img_display, cmap='gray')
        #os.remove("static/d" +str(dt)+".png")
        plt.savefig("static/d" +str(dt)+".png") 

        y_preds = model_vgg16.predict(im, batch_size=batch_size, verbose=1)
        print(y_preds)
        y_prediction = np.argmax(y_preds)
        print('Y Prediction: {}'.format(y_prediction))
        print('Predicted as: {}'.format(activity_map.get('c{}'.format(y_prediction))))

        plt.show()


        # If keyboard interrupt occurs, destroy image 
        # window 
        waitKey(0) 
        #destroyWindow("GeeksForGeeks") 
        sstr=""
        sstr+='''<img src=static/d''' +str(dt)+'''.png width=200 height=200>'''
        print(sstr)
        sstr+='<br>Predicted as: {}'.format(activity_map.get('c{}'.format(y_prediction)))
        return sstr

if __name__ == '__main__':
    app.run(debug=True)