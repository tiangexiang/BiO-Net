import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import importlib
import sys


def argparsing():
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
  parser.add_argument('--valid_dataset', default='monuseg', choices=['monuseg', 'tnbc'], type=str, help='which dataset to validate?')
  parser.add_argument('--backend', default='keras', choices=['keras', 'pytorch'], type=str, help='which backend to use?')
  args = parser.parse_args()

  print()
  print()
  print(args) # print command line args

  return args
  
def main(args, CORE):
    # path verification
    if args.model_path is not None:
      if os.path.isfile(args.model_path):
        print('Model path has been verified.')
      else:
        print('Invalid model path! Please specify a valid model file. Program terminating...')
        exit()

    # pipeline starts
    if not args.evaluate_only:
      CORE.train(args)
    CORE.evaluate(args)
    
if __name__ == '__main__':
  # parse command line args
  args = argparsing()

  # import dependencies
  CORE = importlib.import_module(args.backend+'_version')

  main(args, CORE)
