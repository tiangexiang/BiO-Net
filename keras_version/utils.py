import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # mute deprecation warnings
from keras.optimizers import Adam, SGD
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

import numpy as np
import sys
from PIL import Image
import argparse
from matplotlib import pyplot as plt

from .dataloader import *
from .model import *
from .metrics import *


def train(args):
  # load data
  x_val, y_val = load_data(args.valid_data, args.valid_dataset)
  x_train, y_train = load_data(args.train_data, 'monuseg')
  print('data loading finished.')

  K.clear_session()
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  K.set_learning_phase(1)

  input_shape = x_train[0].shape

  # create model
  model = BiONet(
      input_shape,
      num_classes=args.num_class,
      num_layers=4,
      iterations=args.iter,
      multiplier=args.multiplier,
      integrate=args.integrate
  ).build()
  
  # augmentation
  train_gen = get_augmented(
  x_train, y_train, batch_size=args.batch_size,
  data_gen_args = dict(
      rotation_range=15.,
      width_shift_range=0.05,
      height_shift_range=0.05,
      shear_range=50,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='constant'
  ))

  model.compile(
      optimizer=Adam(lr=args.lr,decay=args.lr_decay), 
      loss = 'binary_crossentropy',
      metrics=[iou, dice_coef]
  )

  print('model successfully built and compiled.')
  
  integrate = '_int' if args.integrate else ''
  weights = '_weights' if args.save_weight else ''
  cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.h5'
  callbacks = [keras.callbacks.ModelCheckpoint("checkpoints/"+args.exp+"/"+cpt_name,monitor='val_iou', mode='max',verbose=0, save_weights_only=args.save_weight, save_best_only=True)]
  if not os.path.isdir("checkpoints/"+args.exp):
    os.mkdir("checkpoints/"+args.exp)
  
  print('\nStart training...')
  history = model.fit_generator(
      train_gen,
      steps_per_epoch=args.steps,
      epochs=args.epochs,
      validation_data=(x_val, y_val),
      callbacks=callbacks
  )
  print('\nTraining fininshed!')

  K.clear_session()

def evaluate(args):
  # load data
  x_val, y_val = load_data(args.valid_data, args.valid_dataset)
  print('data loading finished.')

  K.clear_session()
  K.set_learning_phase(1)
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)

  if args.model_path is None:
    integrate = '_int' if args.integrate else ''
    weights = '_weights' if args.save_weight else ''
    cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.h5'
    model_path = "checkpoints/"+args.exp+"/"+cpt_name
  else:
    model_path = args.model_path
  print('Restoring model from path: '+model_path)

  if args.save_weight:
    model = BiONet(
          input_shape,
          num_classes=args.num_class,
          num_layers=4,
          iterations=args.iter,
          multiplier=args.multiplier,
          integrate=args.integrate
      ).build().load_weights(model_path)
  else:
    model = load_model(model_path, compile=False)

  model.compile(
      optimizer=Adam(lr=args.lr,decay=args.lr_decay), 
      loss='binary_crossentropy',
      metrics=[iou, dice_coef]
  )

  print('\nStart evaluation...')
  result = model.evaluate(x_val,y_val,batch_size=args.batch_size)
  print('Validation loss:\t', result[0])
  print('Validation  iou:\t', result[1])
  print('Validation dice:\t', result[2])

  print('\nEvaluation finished!')

  if args.save_result:

    # save metrics
    if not os.path.exists("checkpoints/"+args.exp+"/outputs"):
      os.mkdir("checkpoints/"+args.exp+"/outputs")

    with open("checkpoints/"+args.exp+"/outputs/result.txt", 'w+') as f:
      f.write('Validation loss:\t'+str(result[0])+'\n')
      f.write('Validation  iou:\t'+str(result[1])+'\n')
      f.write('Validation dice:\t'+str(result[2])+'\n')
    
    print('Metrics have been saved to:', "checkpoints/"+args.exp+"/outputs/result.txt")

    # predict and save segmentations
    results = model.predict(x_val,batch_size=args.batch_size,verbose=1)
    results = (results > 0.5).astype(np.float32) # Binarization. Comment out this line if you don't want to
  
    print('\nPrediction finished!')
    print('Saving segmentations...')

    if not os.path.exists("checkpoints/"+args.exp+"/outputs/segmentations"):
      os.mkdir("checkpoints/"+args.exp+"/outputs/segmentations")

    for i in range(results.shape[0]):
      plt.imsave("checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,0],cmap='gray') # binary segmenation

    print('A total of '+str(results.shape[0])+' segmentation results have been saved to:', "checkpoints/"+args.exp+"/outputs/segmentations/")

  K.clear_session()

def get_augmented(
  X_train, 
  Y_train, 
  X_val=None,
  Y_val=None,
  batch_size=32, 
  seed=0, 
  data_gen_args = dict(
      rotation_range=10.,
      #width_shift_range=0.02,
      height_shift_range=0.02,
      shear_range=5,
      #zoom_range=0.3,
      horizontal_flip=True,
      vertical_flip=False,
      fill_mode='constant'
  )):
  
  # Train data, provide the same seed and keyword arguments to the fit and flow methods
  X_datagen = ImageDataGenerator(**data_gen_args)
  Y_datagen = ImageDataGenerator(**data_gen_args)
  X_datagen.fit(X_train, augment=True, seed=seed)
  Y_datagen.fit(Y_train, augment=True, seed=seed)
  X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
  Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
  
  train_generator = zip(X_train_augmented, Y_train_augmented)

  if not (X_val is None) and not (Y_val is None):
      # Validation data, no data augmentation, but we create a generator anyway
      X_datagen_val = ImageDataGenerator(**data_gen_args)
      Y_datagen_val = ImageDataGenerator(**data_gen_args)
      X_datagen_val.fit(X_val, augment=True, seed=seed)
      Y_datagen_val.fit(Y_val, augment=True, seed=seed)
      X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
      Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

      # combine generators into one which yields image and masks
      val_generator = zip(X_val_augmented, Y_val_augmented)
      
      return train_generator, val_generator
  else:
      return train_generator
