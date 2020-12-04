import os
import sys
sys.path.append(r"/Users/monamiwaki/Fall2020-Workshop3/backend/model")
save_model_path = './model/model.h5'

yourpath = os.path.dirname(os.path.abspath(__file__)) #current filepath

parentpath = os.path.abspath(os.path.join(yourpath, os.pardir))
sys.path.append(parentpath)

from model.model import ImageDetectModel
from PIL import Image
import matplotlib
from keras.models import load_model
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import img_to_array
import os
from PIL import ImageFilter
import tensorflow as tf 
import numpy as np 
size = (28, 28)

weights = "imagenet"
base_model = VGG16(weights=weights)


model = Model(base_model.input, base_model.get_layer('fc1').output)
graph = tf.get_default_graph()


class ImageDetectModel:
    def __init__(self):
        model = Model(base_model.input, base_model.get_layer('fc1').output)
        self.model = model.load_weights(save_model_path) 


    def pred(self, request):
        f = request.files['image']
        img = Image.open(f)
        
        
        #img = img.convert('L')
        
        # RESIZING IMAGES 
        height = 224
        width = 224
        dim = (width, height)
        img = img.resize(dim)
    
        # GAUSSIAN BLUR 
        blurred = img.filter(ImageFilter.GaussianBlur(radius = 5))

        img = image.img_to_array(blurred) 
        img = np.expand_dims (img, axis = 0)

        with graph.as_default():
            return model.predict(img)[0][0]
