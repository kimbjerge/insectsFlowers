# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:03:46 2024

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

#from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

def createConvNext(input_shape, number_of_classes, trainable):
    
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
    
        conv_base = tf.keras.applications.ConvNeXtBase(
            #model_name="convnext_base",
            include_top=False,
            include_preprocessing=True,
            weights= "imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000
        )
        
        for layer in conv_base.layers: #[-10:]:  # Unfreeze the last 10 layers
            layer.trainable = trainable        
        conv_base.trainable = trainable
    
        dropout_rate = 0.2
        model = models.Sequential()
       
        model.add(conv_base)
        model.add(layers.GlobalMaxPooling2D(name="gap"))
        #model.add(layers.Flatten(name="flatten"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name="dropout_out"))
        #model.add(layers.Dense(256, activation='relu', name="fc1"))
        model.add(layers.Dense(number_of_classes, activation="softmax", name="fc_out"))
        
        if trainable:
            learning_rate = 0.00001 # Learning rate for finetuning
        else:
            learning_rate = 0.0001 
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        model.summary()

        print("Learnable parameters:", model.count_params(), "learning rate", learning_rate)
        
    return model

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
    
        for layer in conv_base.layers: #[-10:]:  # Unfreeze the last 10 layers
            layer.trainable = trainable        
        conv_base.trainable = trainable
    
        dropout_rate = 0.2
        model = models.Sequential()

        """ Alternative creation
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

        if trainable:
            learning_rate = 0.00001 # Learning rate for finetuning
        else:
            learning_rate = 0.0001 
        
        model.compile(
            #optimizer=optimizers.RMSprop(lr=2e-5),
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        model.summary()

        print("Learnable parameters:", model.count_params(), "learning rate", learning_rate)
        
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
  
        """ Alternative output layers
        # Load the ResNet50 model with pre-trained weights
        base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the base model layers (optional for fine-tuning)
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers on top of ResNet50
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        predictions = layers.Dense(number_of_classes, activation='softmax')(x)

        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        """

        baseModel = tf.keras.applications.ResNet50V2(include_top=False, 
                                                  weights= "imagenet", #None
                                                  input_tensor=None, 
                                                  input_shape=input_shape,
                                                  pooling=None, #max
                                                  classes=number_of_classes,
                                                  classifier_activation="softmax")
        # Freeze layers for transfer learning
        for layer in baseModel.layers: #[-10:]:  # Unfreeze the last 10 layers
            layer.trainable = trainable
            
        # Used when using weights = "imagenet"
        headModel = baseModel.output
        headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(256, activation="relu")(headModel)
        headModel = layers.Dropout(0.5)(headModel)
        headModel = layers.Dense(number_of_classes, activation="softmax")(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        if trainable:
            learning_rate = 0.00001 # Learning rate for finetuning
        else:
            learning_rate = 0.0001 

        model.compile(
            #optimizer="adam", 
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        model.summary()
    
        print("Learnable parameters:", model.count_params(), "learning rate", learning_rate)

    return model    

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

if __name__=='__main__': 
    
    parser = argparse.ArgumentParser()
    
    # Arguments to be changed 
    parser.add_argument('--modelType', default='EfficientNetB4') # Model to be trained EfficientNetB4, ResNet50v2, ConvNeXtBase
    parser.add_argument('--dataDir', default='../datasets/NI2-19cls') # Path to dataset
    parser.add_argument('--epochs', default='100', type=int) # Training epochs
    parser.add_argument('--patience', default='5', type=int) # Patience epochs before early stopping (Min. validation loss)   
    parser.add_argument('--batch', default='32', type=int) # Batch size
    parser.add_argument('--trainBaseLayers', default='', type=bool) # Default false when no parameter (finetune base layers of model)
    parser.add_argument('--imageRescaling', default='', type=bool) # Default image rescaling False (Rescaling then multiply pixels with 1.0/255)
    parser.add_argument('--modelName', default='', type=str) # If modelName given then model will be finetuned and trainBaseLayers will be set to True

    args = parser.parse_args()
    
    # Directory with subdirectories for each class with cropped images in jpg format
    #data_dir = '../datasets/NI2-19cls'
    #data_dir = '../../data/NI2-19cls'
    data_dir = args.dataDir
    
    # Directory for saving h5 models for each run
    models_dir = './models_save'   
    log_dir = './hparam_tuning19cls'
    
    modelType = args.modelType
    #modelType = "EfficientNetB4"
    #modelType = "ResNet50v2"
    #modelType = "ConvNeXtBase"

    finetuneName = ""
    if args.modelName != "":
        args.trainBaseLayers = True
        finetuneName = "-Finetuned"
    
    base_layers_trainable = args.trainBaseLayers    
    epochs = args.epochs
    batch_size = args.batch

    image_size = 224 # MobileNetV2, EfficientNetB4, ResNet50V2, ConvNeXtBase
    #image_size = 299 # InceptionV3
    number_of_classes = 19
    
    NUM_DATA = 13817 # 19 classes (13628)
    TEST_SPLIT = 0.2
    NUM_TRAIN = NUM_DATA*(1.0 - TEST_SPLIT)
    NUM_TEST = NUM_DATA*TEST_SPLIT

    input_shape= (image_size, image_size, 3)

    if modelType == "ResNet50v2":
        model = createResNetV2(input_shape, number_of_classes, base_layers_trainable)
    if modelType == "EfficientNetB4":
        model = createEfficientNet(input_shape, number_of_classes, base_layers_trainable)
    if modelType == "ConvNeXtBase":
        model = createConvNext(input_shape, number_of_classes, base_layers_trainable)
        
    train_generator, validation_generator = createDataGenerators(data_dir, image_size, batch_size, imageRescaling=args.imageRescaling)

    if args.imageRescaling:    
        best_model_name = models_dir + '/' +  modelType + '-19cls-' + str(epochs) + '-ExtRescaled' + finetuneName + '.keras'
    else:
        best_model_name = models_dir + '/' +  modelType + '-19cls-' + str(epochs) + '-Ext' + finetuneName + '.keras'
        
    print("Saving best model as", best_model_name)
        
    myCallbacks = [
        tf.keras.callbacks.TensorBoard(log_dir),
        ModelCheckpoint(best_model_name, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    ]
    
    if args.modelName != "": 
        print("Finetuning model loading weights", args.modelName)
        model.load_weights(models_dir + '/' + args.modelName)
 
    print(args)
    
    #history = model.fit_generator(
    history = model.fit(
        train_generator,
        #steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        #validation_steps=validation_generator.samples // batch_size,
        #workers=8,
        #verbose=1,
        callbacks=myCallbacks
        #use_multiprocessing=True,
    )
    
    model = tf.keras.models.load_model(best_model_name)
    print('Model predict')
    Y_pred = model.predict(validation_generator) #, 173//batch_size+1
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(classification_report(validation_generator.classes, y_pred))
    report = classification_report(validation_generator.classes, y_pred, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    print('F1-score:', f1_score)

    conf = confusion_matrix(validation_generator.classes, y_pred, normalize='true')
    conf = np.round(conf*100)
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conf, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(modelType + '-Confmatrix-19cls' + finetuneName + '.png')
    plt.close(figure)

