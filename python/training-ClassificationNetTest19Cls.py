# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:48:46 2022

@author: Kim Bjerge
    
    Training insect classifier with ResNet or EfficientNet
"""

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.

import io
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

## LIMIT MEMORY - Can be uncommented
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)


from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from sklearn.metrics import classification_report, confusion_matrix

def createEfficientNet(input_shape, number_of_classes, trainable):
    
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:
        print("Not connected to a TPU runtime. Using CPU/GPU strategy")
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():   # loading pretrained conv base model
    
        conv_base = tf.keras.applications.EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape)    
    
        #for layer in conv_base.layers:
        #    layer.trainable = trainable        
        conv_base.trainable = trainable
    
        dropout_rate = 0.2
        model = models.Sequential()

        """
        headModel = conv_base.output
        headModel = layers.GlobalMaxPooling2D(name="gap")(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(256, activation='relu', name="fc1")(headModel)
        if dropout_rate > 0:
         headModel = layers.Dropout(dropout_rate, name="dropout_out")(headModel)
        headModel = layers.Dense(number_of_classes, activation="softmax", name="fc_out")(headModel)
        model = Model(inputs=conv_base.input, outputs=headModel)
        """
        
        model.add(conv_base)
        model.add(layers.GlobalMaxPooling2D(name="gap"))
        #model.add(layers.Flatten(name="flatten"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name="dropout_out"))
        #model.add(layers.Dense(256, activation='relu', name="fc1"))
        model.add(layers.Dense(number_of_classes, activation="softmax", name="fc_out"))
        #model.add(layers.Dense(number_of_classes, activation="sigmoid", name="fc_out"))
        
        model.compile(
            #optimizer=optimizers.RMSprop(lr=2e-5),a
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        #model.compile(
        #    optimizer="adam", 
        #    loss="categorical_crossentropy", 
        #    metrics=["accuracy"]
        #)
        model.summary()
        
    return model

def createResNetV2(input_shape, number_of_classes, trainable):
    
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:
        print("Not connected to a TPU runtime. Using CPU/GPU strategy")
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
  
        baseModel = tf.keras.applications.ResNet50V2(include_top=False, 
                                                  weights= "imagenet", #None
                                                  input_tensor=None, 
                                                  input_shape=input_shape,
                                                  pooling=None, #max
                                                  classes=number_of_classes,
                                                  classifier_activation="softmax")
        # Freeze layers for transfer learning
        for layer in baseModel.layers:
            layer.trainable = trainable
            
        # Used when using weights = "imagenet"
        headModel = baseModel.output
        headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(256, activation="relu")(headModel)
        headModel = layers.Dropout(0.5)(headModel)
        headModel = layers.Dense(number_of_classes, activation="softmax")(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        model.summary()
    
        print("Learnable parameters:", model.count_params())

    return model    

def createDataGenerators(data_dir, image_size, batch_size, modelType, seed = 1):
    
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
    if modelType == "Efficient":
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
    else:
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

if __name__=='__main__': 
    
    # Directory with subdirectories for each class with cropped images in jpg format
    #data_dir = '../data/NI2'
    data_dir = '../data/NI2-19cls'
    # Directory for saving h5 models for each run
    models_dir = './models_save'   
    log_dir = './hparam_tuning19cls'
    
    modelType = "Efficient"
    #modelType = "ResNet"
  
    number_of_classes = 19
    #number_of_classes = 11
    batch_size = 32
    epochs = 100 # 38 EfficientNetB0+others, 27 MobileNetV2, 25 InceptionV3, 35 ResNet50V2, 25 ResNet50V2 with imagenet
    image_size = 224 # MobileNetV2, EfficientNetB0, ResNet50V2
    #image_size = 299 # InceptionV3
    
    #NUM_DATA = 7360
    #NUM_DATA = 5745 # 10 classes
    NUM_DATA = 13825 # 19 classes (13628)
    TEST_SPLIT = 0.2
    NUM_TRAIN = NUM_DATA*(1.0 - TEST_SPLIT)
    NUM_TEST = NUM_DATA*TEST_SPLIT

    input_shape= (image_size, image_size, 3)

    if modelType == "ResNet":
        model = createResNetV2(input_shape, number_of_classes, False)
    else:
        model = createEfficientNet(input_shape, number_of_classes, False)

    train_generator, validation_generator = createDataGenerators(data_dir, image_size, batch_size, modelType)

    # Extend with examed hyperparameters
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
    HP_IMG_SIZE = hp.HParam('image_size', hp.Discrete([224]))

    hparams = {
            HP_BATCH_SIZE: batch_size,
            HP_IMG_SIZE: image_size
            }
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=NUM_TEST // batch_size,
        verbose=1,
        callbacks=[
                   tf.keras.callbacks.TensorBoard(log_dir),
                   hp.KerasCallback(log_dir, hparams),
                   ],
        #use_multiprocessing=True,
        #workers=4,
    )
    
    print('Model predict')
    Y_pred = model.predict_generator(validation_generator) #, 173//batch_size+1
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(classification_report(validation_generator.classes, y_pred))
    report = classification_report(validation_generator.classes, y_pred, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    print('F1-score:', f1_score)

    if modelType == "ResNet": 
        model.save(models_dir + '/' +  'ResNet50v2-T35.h5')
    else:
        model.save(models_dir + '/' +  'EfficientNetB4-softmax-19cls-100.h5')

    conf = confusion_matrix(validation_generator.classes, y_pred, normalize='true')
    conf = np.round(conf*100)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conf, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confmatrix-19cls.png')
    plt.close(figure)

