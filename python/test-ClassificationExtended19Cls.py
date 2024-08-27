# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:56:46 2024

@author: Kim Bjerge
    
    Training insect classifier with ResNet50V2, EfficientNetB4, ConvNeXtBase
"""

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

## LIMIT MEMORY - Can be uncommented
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def createDataGenerators(data_dir, image_size, batch_size, imageRescaling=False, seed = 1):
    
    # Train data genrator
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode="nearest",
        validation_split=0.2
    )
    """
    if imageRescaling:
        # Settings from AMT
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range = 180,
            horizontal_flip = True,
            vertical_flip = True,
            zoom_range=0.3,
            validation_split=0.2,
            brightness_range=[0.9, 1.1]
        )
    
        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2
        )
    else: # EfficientNetB4 used in pipeline (No rescaling)
        # Settings from AMT
        train_datagen = ImageDataGenerator(
            rotation_range = 180,
            horizontal_flip = True,
            vertical_flip = True,
            zoom_range=0.3,
            validation_split=0.2,
            brightness_range=[0.9, 1.1]
        )
    
        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(
            validation_split=0.2
        )
    
    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        data_dir,
        # All images will be resized to target height and width.
        target_size=(image_size, image_size),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=seed
    )
    
    validation_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        shuffle=False,
        seed=seed
    )
    
    return train_generator, validation_generator

def myprint(s):
    with open('modelsummaries.txt','a') as f:
        print(s, file=f)

if __name__=='__main__': 
    
    parser = argparse.ArgumentParser()
    
    # Arguments to be changed 
    #parser.add_argument('--modelType', default='EfficientNetB4') # Model to be trained EfficientNetB4, ResNet50v2, ConvNeXtBase
    parser.add_argument('--dataDir', default='../datasets/NI2-19cls') # Path to dataset
    parser.add_argument('--modelName', default='EfficientNetB4-19cls-80.keras') # Name of model weights
    parser.add_argument('--batch', default='32', type=int) # Batch size
    parser.add_argument('--imageRescaling', default='', type=bool) # Default image rescaling False (Multiply pixels with 1.0/255)
        
    args = parser.parse_args()
        
    # Directory with subdirectories for each class with cropped images in jpg format
    #data_dir = '../datasets/NI2-19cls'
    #data_dir = '../../data/NI2-19cls'
    data_dir = args.dataDir
    # Directory for saving h5 models for each run
    models_dir = './models_save'   
    log_dir = './hparam_tuning19cls'
    
    #modelType = args.modelType
    #modelType = "EfficientNetB4"
    #modelType = "ResNet50v2"
    #modelType = "ConvNeXtBase"
    
    batch_size = args.batch

    image_size = 224 # MobileNetV2, EfficientNetB4, ResNet50V2, ConvNeXtBase
    #image_size = 299 # InceptionV3
    number_of_classes = 19
    
    NUM_DATA = 13817 # 19 classes (13628)
    TEST_SPLIT = 0.2
    NUM_TRAIN = NUM_DATA*(1.0 - TEST_SPLIT)
    NUM_TEST = NUM_DATA*TEST_SPLIT

    input_shape= (image_size, image_size, 3)

    train_generator, validation_generator = createDataGenerators(data_dir, image_size, batch_size, imageRescaling=args.imageRescaling)

    model = tf.keras.models.load_model(models_dir + '/' +  args.modelName)
    
    model.summary()
    
    model.summary(print_fn=myprint)
    with open('modelsummaries.txt','a') as f:
        f.write(args.modelName + '\n')

    print(args)
    
    print('Model predict')
    Y_pred = model.predict(validation_generator) 
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    class_report = classification_report(validation_generator.classes, y_pred)
    print(class_report)
    with open('modelsummaries.txt','a') as f:
        f.write('Confusion Matrix\n')
        f.write(class_report)
        
    report = classification_report(validation_generator.classes, y_pred, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
        
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1_score)
    
    with open("classifiers.txt", "a") as myfile:
        modelName = args.modelName.split('-')[0]
        str = "%s & ? & %0.3f & %0.3f & %0.3f \\\\\n" % (modelName, precision, recall, f1_score)
        myfile.write(str)
        print(str)
    
    conf = confusion_matrix(validation_generator.classes, y_pred, normalize='true')
    conf = np.round(conf*100)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conf, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(args.modelName.split('.')[0] + '_confmatrix_test.png')
    plt.close(figure)

