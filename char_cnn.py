#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: EN_Char_CNN_Text_Classification
# @File   : char_cnn.py
# @Author : Origin.H
# @Date   : 2018/1/9
import tensorflow as tf
import numpy as np
from collections import namedtuple
import logging
import time
import os
import config
import pretraining


class CharConvNet(object):
    conv_pool_layer_hyperparameters = namedtuple('Conv_Pool_Layer_Hyperparameters', ['filter_height', 'out_channel',
                                                                                     'pool_height'])

    def __init__(self, conv_pool_layers, fully_connected_layers, length0, n_class,
                 batch_size=128, learning_rate=0.01, decay_steps=1000, decay_rate=0.98, grad_clip=5, max_to_keep=5):
        self.preparation = pretraining.PreparationForTraining()
        self.conv_pool_layers = []
        for filter_height, out_channel, pool_height in conv_pool_layers:
            self.conv_pool_layers.append(self.conv_pool_layer_hyperparameters(filter_height, out_channel,
                                                                              pool_height))
        self.fully_connected_layers = fully_connected_layers
        self.length0 = length0
        self.n_class = n_class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.grad_clip = grad_clip
        self.max_to_keep = max_to_keep
        self.input_x = self.input_y = self.dropout_keep_prob = None
        self.logit = self.prediction = self.global_step = self.loss = self.optimizing = self.accuracy = None
        self.saver = self.summary_op = None

    def build_graph(self):
        # 展开新图
        tf.reset_default_graph()

        with tf.name_scope("Input_Layer"):
            self.input_x = tf.placeholder(
                tf.float32, shape=[None, self.length0, self.preparation.size_of_alphabet, 1], name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=[None, self.n_class], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        x = self.input_x
        for layer_index, parameters in enumerate(self.conv_pool_layers, start=1):
            with tf.name_scope("Conv_Layer%d" % layer_index), tf.variable_scope("Conv_Layer%d" % layer_index):
                filter_width = x.get_shape()[2].value
                in_channel = x.get_shape()[3].value
                filter_shape = [parameters.filter_height, filter_width, in_channel, parameters.out_channel]
                # 卷积
                conv = tf.nn.conv2d(x, tf.get_variable(
                    'weight', filter_shape, initializer=tf.random_normal_initializer(mean=0, stddev=0.05)
                ), [1, 1, 1, 1], "VALID", name='conv')
                x = tf.nn.bias_add(conv, tf.get_variable(
                    'biases', [parameters.out_channel], initializer=tf.zeros_initializer))
                # x = tf.nn.relu(x, name='relu')

            with tf.name_scope("Pool_Layer%d" % layer_index):
                x = tf.nn.max_pool(x, ksize=[1, parameters.pool_height, 1, 1],
                                   strides=[1, parameters.pool_height, 1, 1], padding='VALID', name='pool')

        with tf.name_scope("Reshape_Layer"):
            dims = x.get_shape()[1].value * x.get_shape()[2].value * x.get_shape()[3].value
            x = tf.reshape(x, [-1, dims])

        for layer_index, n_out in enumerate(self.fully_connected_layers, start=1):
            with tf.name_scope("FC_Layer%d" % layer_index), tf.variable_scope("FC_Layer%d" % layer_index):
                n_in = x.get_shape()[-1].value
                x = tf.nn.xw_plus_b(x, tf.get_variable(
                    'weight', [n_in, n_out], initializer=tf.random_normal_initializer), tf.get_variable(
                    'biases', [n_out], initializer=tf.zeros_initializer), name='fc')
                # x = tf.nn.relu(x, name='relu')

            with tf.name_scope("Dropout_Layer%d" % layer_index):
                x = tf.nn.dropout(x, self.dropout_keep_prob, name='dropout')

        with tf.name_scope("Softmax_Layer"), tf.variable_scope("Softmax_Layer"):
            self.logit = tf.nn.xw_plus_b(
                x,
                tf.get_variable(
                    'weight', [x.get_shape()[-1].value, self.n_class], initializer=tf.random_normal_initializer
                ),
                tf.get_variable('biases', [self.n_class], initializer=tf.zeros_initializer),
                name='logit'
            )
            self.prediction = tf.nn.softmax(self.logit, name='softmax_out')

        with tf.name_scope('Train_ops'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logit),
                                       name='loss')
            # 梯度截断
            # variables = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, variables), self.grad_clip)
            # opt = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer = opt.apply_gradients(
            #     zip(grads, variables), global_step=self.global_step, name='optimizer')

            # 学习率衰减
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                       self.decay_rate, name='learning_rate_decay')
            opt = tf.train.AdamOptimizer(learning_rate, name='optimizer')
            grads_and_vars = opt.compute_gradients(self.loss)
            self.optimizing = opt.apply_gradients(grads_and_vars, self.global_step, 'optimizing')

            correct_predictions = tf.equal(tf.argmax(self.logit, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            self.saver = tf.train.Saver(max_to_keep=self.max_to_keep, name='saver')

            loss_summary = tf.summary.scalar('loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            grad_summaries = []
            for grad, var in grads_and_vars:
                if grad is None:
                    grad = 0.0
                grad_summaries.append(tf.summary.histogram('gradient of %s' % var.name, grad))
            gradients_merged = tf.summary.merge(grad_summaries)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary, gradients_merged], name='summary_op')

    def fit(self, training_path, checkpoint_path, keep_prob=0.5, epochs=20,
            display_steps=1000, save_steps=1000, name='my_train'):

        # 训练
        # training_path: 训练文本所在的文件夹
        # checkpoint_path: 参数保存的文件夹
        # name: 训练的命名

        self.build_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                checkpoint = tf.train.latest_checkpoint(checkpoint_path)
                self.saver.restore(sess, checkpoint)
                logging.info(config.CHECKPOINT_RESTORE_MESS % checkpoint)
                print(config.CHECKPOINT_RESTORE_MESS % checkpoint)
            except ValueError:
                logging.info(config.CHECKPOINT_RESTORE_FAIL_MESS)
                print(config.CHECKPOINT_RESTORE_FAIL_MESS)
            start_time = time.time()
            summary_writer = tf.summary.FileWriter(os.path.join(config.SUMMARY_HOME, (config.SUMMARY_FILE % name)),
                                                   sess.graph)
            for epoch in range(epochs):
                for x_batch, y_batch in self.preparation.get_batch_from_file(
                        training_path, self.batch_size, self.length0, self.n_class):
                    train_ops = [self.loss, self.optimizing, self.accuracy, self.global_step, self.summary_op]
                    feed_dict = {self.input_x: x_batch,
                                 self.input_y: y_batch,
                                 self.dropout_keep_prob: keep_prob}
                    batch_loss, _, batch_accuracy, global_step, summaries = sess.run(train_ops, feed_dict)
                    if global_step % display_steps == 0:
                        end_time = time.time()
                        logging.info(config.DISPLAY_STEPS_MESS % (
                            epoch + 1, global_step, batch_loss, batch_accuracy * 100, end_time - start_time))
                        print(config.DISPLAY_STEPS_MESS % (
                            epoch + 1, global_step, batch_loss, batch_accuracy * 100, end_time - start_time))
                        start_time = time.time()

                    if global_step % save_steps == 0:
                        self.saver.save(sess, os.path.join(checkpoint_path, (config.MODEL_FILE % name)),
                                        global_step=global_step)
                        summary_writer.add_summary(summaries, global_step)

    def validate(self, testing_path, checkpoint_path, batch_size=512):
        # 验证
        # testing_path: 验证集文本所在的文件夹
        # checkpoint_path: 参数保存的文件夹
        # batch_size: 验证时batch的大小

        self.build_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                checkpoint = tf.train.latest_checkpoint(checkpoint_path)
                self.saver.restore(sess, checkpoint)
                logging.info(config.CHECKPOINT_RESTORE_MESS % checkpoint)
                print(config.CHECKPOINT_RESTORE_MESS % checkpoint)
            except ValueError:
                logging.info(config.CHECKPOINT_RESTORE_FAIL_MESS)
                print(config.CHECKPOINT_RESTORE_FAIL_MESS)
                return 1

            accuracies = []
            for x_batch, y_batch in self.preparation.get_batch_from_file(testing_path, batch_size, self.length0,
                                                                         self.n_class):
                feed_dict = {self.input_x: x_batch,
                             self.input_y: y_batch,
                             self.dropout_keep_prob: 1.0}
                batch_accuracy = sess.run([self.accuracy], feed_dict)
                accuracies.append(batch_accuracy[0])
            accuracy = np.array(accuracies).mean()
            logging.info(config.DISPLAY_TEST_MESS % (accuracy * 100))
            print(config.DISPLAY_TEST_MESS % (accuracy * 100))
