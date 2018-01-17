#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: EN_Char_CNN_Text_Classification
# @File   : validate.py
# @Author : Origin.H
# @Date   : 2018/1/16
import os
import logging
import char_cnn
import config
import hyperparameters


if __name__ == "__main__":
    test = 'ag_news_test'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y %b %d %H:%M:%S',
                        filename=os.path.join(config.LOG_HOME, (config.LOG_FILE % test)),
                        filemode='a')
    conv_pool_layers = [
        (7, 256, 3),
        (7, 256, 3),
        (3, 256, 1),
        (3, 256, 1),
        (3, 256, 1),
        (3, 256, 3)
    ]
    fully_connected_layers = [
        1024,
        1024
    ]
    model = char_cnn.CharConvNet(conv_pool_layers=conv_pool_layers, fully_connected_layers=fully_connected_layers,
                                 length0=hyperparameters.LENGTH0, n_class=hyperparameters.N_CLASS,
                                 batch_size=hyperparameters.BATCH_SIZE, learning_rate=hyperparameters.LEARNING_RATE,
                                 decay_steps=hyperparameters.DECAY_STEPS, decay_rate=hyperparameters.DECAY_RATE,
                                 grad_clip=hyperparameters.GRAD_CLIP, max_to_keep=config.MAX_TO_KEEP)
    model.validate(config.TEST_TEXT_HOME, config.CHECKPOINT_HOME, 1024)
    logging.info(config.DONE_TESTING_MESS)
    print(config.DONE_TESTING_MESS)
