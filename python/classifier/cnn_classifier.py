import six
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
from skimage import io
from skimage.transform import resize
import numpy as np
from tensorflow.keras import backend as K
#import keras
#from skimage import io
#from skimage import color
#from skimage.transform import resize
#from PIL import Image
#import scipy
#from pathlib import Path

class CnnClassifier:
    def __init__(self, modeltype, species):
        self.modeltype = modeltype
        self.model = self.loadmodel()
        self.species = species
        self.dim = (224, 224)
 
    def loadmodel(self):
        model = tf.keras.models.load_model(self.modeltype + '.h5')
        print('Loaded model: ', self.modeltype + '.h5')
        model.summary()
        return model

    def makePrediction(self, im, number=1):
        resizedImg = cv2.resize(im, self.dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ', resizedImg.shape)
        rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_RGB2BGR)
        img = tf.keras.preprocessing.image.img_to_array(rgbImg)
        img = np.expand_dims(img, axis = 0)
        
        predictions = self.model.predict(img, verbose=1)
        predictions = predictions[0]
        probability = np.amax(predictions)
        index = np.where(predictions == probability)
        idx = index[0][0]
        print(predictions, probability, idx, self.species[idx])
        return idx, self.species[idx], probability 

    def get_activations(self, x, layer, batch_size=128):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.
        :param x: Input for computing the activations.
        :type x: `np.ndarray`. Example: x.shape = (80, 80, 3)
        :param model: pre-trained Keras model. Including weights.
        :type model: keras.engine.sequential.Sequential. Example: model.input_shape = (None, 80, 80, 3)
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`. Example: layer = 'flatten_2'
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`. Example: activations.shape = (1, 2000)
        """
    
        #print(layer)
        
        layer_names = [layer.name for layer in self.model.layers]
        if isinstance(layer, six.string_types):
            if layer not in layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(layer_names) - 1))
            layer_name = layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')
    
        print(layer_name)
        layer_output = self.model.get_layer(layer_name).output
        layer_input = self.model.input
        output_func = K.function([layer_input], [layer_output])
    
        # Apply preprocessing
        if x.shape == K.int_shape(self.model.input)[1:]:
            x_preproc = np.expand_dims(x, 0)
        else:
            x_preproc = x
        assert len(x_preproc.shape) == 4
    
        # Determine shape of expected output and prepare array
        output_shape = output_func([x_preproc[0][None, ...]])[0].shape
        activations = np.zeros((x_preproc.shape[0],) + output_shape[1:], dtype=np.float32)
    
        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preproc.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preproc.shape[0])
            activations[begin:end] = output_func([x_preproc[begin:end]])[0]
    
        return activations

    def makePredictionLayerOut(self, im, number=1):
        resizedImg = cv2.resize(im, self.dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ', resizedImg.shape)
        rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_RGB2BGR)
        img = tf.keras.preprocessing.image.img_to_array(rgbImg)
        img = np.expand_dims(img, axis = 0)
        
        predictions = self.model.predict(img, verbose=1)
        predictions = predictions[0]
        probability = np.amax(predictions)
        index = np.where(predictions == probability)
        idx = index[0][0]
                
        #activations = self.get_activations(img, "dropout_out")
        activations = self.get_activations(img, "fc_out")
        print(activations)
        
        #print(predictions, probability, idx, self.species[idx])
        return idx, self.species[idx], activations
    
    def makePredictionFile(self, imgFile):
        img = io.imread(imgFile)
        img = resize(img, self.dim, anti_aliasing = True)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_pre = np.expand_dims(img, axis = 0)
        
        predictions = self.model.predict(img_pre)
        #print(decode_predictions(predictions, top=2)[0])
        predictions = predictions[0]
        probability = np.amax(predictions)
        index = np.where(predictions == probability)
        idx = index[0][0]
        #print(predictions, probability, idx, self.species[idx])
        print(predictions, self.species[idx])
        return idx, self.species[idx]