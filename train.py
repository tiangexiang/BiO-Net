import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import sys
from PIL import Image
import argparse

from keras.models import Model, load_model
from keras.layers import multiply, add, Permute, Reshape, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, concatenate, Add, Concatenate
from keras import backend as K
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # mute deprecation warnings

import keras
from keras.optimizers import Adam, SGD
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

from core import *
from utils import get_augmented

from matplotlib import pyplot as plt

def train(args, train_data, val_data):
  x_train, y_train = train_data[0], train_data[1]
  x_val, y_val = val_data[0], val_data[1]

  K.clear_session()
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  K.set_learning_phase(1)

  input_shape = x_train[0].shape

  #create model
  model = BiONet(
      input_shape,
      num_classes=args.num_class,
      num_layers=4,
      iterations=args.iter,
      multiplier=args.multiplier,
      integrate=args.integrate
  ).build()
  

  #augmentation
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
  
  print('Start training...')
  history = model.fit_generator(
      train_gen,
      steps_per_epoch=args.steps,
      epochs=args.epochs,
      validation_data=(x_val, y_val),
      callbacks=callbacks
  )
  print('Training fininshed!')

  K.clear_session()
  
  return model

def evaluate(args, valid_data):
  print('Start evaluation...')

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
      optimizer=Adam(lr=0.01,decay=4e-6), 
      loss='binary_crossentropy',
      metrics=[iou, dice_coef]
  )

  x, y = valid_data[0], valid_data[1]
  result = model.evaluate(x,y,batch_size=args.batch_size)
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
    results = model.predict(x,batch_size=args.batch_size,verbose=1)
    results = (results > 0.5).astype(np.float32) # Binarization. Comment out this line if you don't want to
  
    print('\nPrediction finished!')
    print('Saving segmentations...')

    if not os.path.exists("checkpoints/"+args.exp+"/outputs/segmentations"):
      os.mkdir("checkpoints/"+args.exp+"/outputs/segmentations")

    for i in range(results.shape[0]):
      plt.imsave("checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,0],cmap='gray') # binary segmenation

    print('A total of '+str(results.shape[0])+' segmentation results have been saved to:', "checkpoints/"+args.exp+"/outputs/segmentations/")

  K.clear_session()
  
def main():
    parser = argparse.ArgumentParser(description='BiO-Net')
    parser.add_argument('--epochs', default=300, type=int, help='trining epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--steps', default=250, type=int, help='steps per epoch')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=3e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_class', default=1, type=int, help='model output channel number')
    parser.add_argument('--multiplier', default=1.0, type=float, help='parameter multiplier')
    parser.add_argument('--iter', default=1, type=int, help='recurrent iteration')
    parser.add_argument('--integrate', action='store_true', help='integrate all inferenced features')
    parser.add_argument('--save_weight', action='store_true', help='save weight only')
    parser.add_argument('--train_data', default='./data/train', type=str, help='data path')
    parser.add_argument('--valid_data', default='./data/valid', type=str, help='data path')
    parser.add_argument('--exp', default='1', type=str, help='experiment number')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate only?')
    parser.add_argument('--save_result', action='store_true', default=False, help='save results to exp folder?')
    parser.add_argument('--model_path', default=None, type=str, help='path to model check')
    args = parser.parse_args()

    print()
    print()
    print(args) # print command line arguments

    # path verification
    if args.model_path is not None:
      if os.path.isfile(args.output_path):
        print('Model path has been verified.')
      else:
        print('Invalid model path! Please specify a valid model file. Program terminating...')
        exit()

    valid_x, valid_y = load_data(args.valid_data, 'monuseg')
    # valid_x, valid_y = load_data(args.valid_data, 'tnbc')
    # uncomment above to validate with tnbc,
    # NOTE: tnbc data path must be specified in --valid_data
    val_data = [valid_x,valid_y]

    if not args.evaluate_only:
      train_x, train_y = load_data(args.train_data, 'monuseg')
      train_data = [train_x, train_y]
      print('data loading finish')

      train(args,train_data,val_data)

    evaluate(args, val_data)
    
if __name__ == '__main__':
    main()
  
