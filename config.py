#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: EN_Char_CNN_Text_Classification
# @File   : config.py
# @Author : Origin.H
# @Date   : 2018/1/9

# Directories
PROJECT_HOME = r'/home/dp/python/EN_Char_CNN_Text_Classification'
TRAIN_TEXT_HOME = PROJECT_HOME + r'/TrainText'
TEST_TEXT_HOME = PROJECT_HOME + r'/TestText'
CHECKPOINT_HOME = PROJECT_HOME + r'/CheckPoints'
SUMMARY_HOME = PROJECT_HOME + r'/Summaries'
LOG_HOME = PROJECT_HOME + r'/Logs'
# Files
MODEL_FILE = 'ENCharCNNTextClassification_%s'
SUMMARY_FILE = 'ENCharCNNTextClassification_%s'
LOG_FILE = 'Logs_of_%s.log'
# Messages
# pretraining.py
ENCODE_SUCCESS_MESS = 'One-Hot encoding done! Totally %d words have been skipped.'
START_TRAIN_FILE_MESS = 'Start to train from file: %s.'
DONE_TRAIN_FILE_MESS = 'Finish to train from file: %s.'
OPEN_DIR_MESS = 'Open directory: %s.'
# char_cnn.py
CHECKPOINT_RESTORE_MESS = 'Checkpoint: %s has been restored.'
CHECKPOINT_RESTORE_FAIL_MESS = 'No checkpoints being restored.'
DISPLAY_STEPS_MESS = 'Epochs: %d, total steps: %d, batch cost: %.4f, batch accuracy: %.2f%%, time to use: %d seconds.'
DISPLAY_TEST_MESS = 'Accuracy: %.2f%%.'
# train.py
DONE_TRAINING_MESS = 'The training has been done!'
# validate.py
DONE_TESTING_MESS = 'The testing has been done!'
# train options
MAX_TO_KEEP = 10
EPOCHS = 55
DISPLAY_STEPS = 1000
SAVE_STEPS = 5000
