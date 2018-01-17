#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: EN_Char_CNN_Text_Classification
# @File   : pretraining.py
# @Author : Origin.H
# @Date   : 2018/1/11
import numpy as np
import os
import re
import csv
import config


class PreparationForTraining:

    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
        self.size_of_alphabet = len(self.alphabet)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.alphabet))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        self.character_skipped = 0

    def x_one_hot_encode_by_alphabet(self, text, length0):
        assert isinstance(text, str)
        text = text.lower()
        x_input = np.zeros([length0, self.size_of_alphabet], dtype=np.float32)
        try:
            for index, char in enumerate(text):
                if char in self.char_to_index:
                    x_input[index, self.char_to_index[char]] = 1
                else:
                    self.character_skipped += 1
                    pass
        except IndexError:
            pass
        # logging.info(config.ENCODE_SUCCESS_MESS % self.character_skipped)
        # print(config.ENCODE_SUCCESS_MESS % self.character_skipped)
        return x_input

    @staticmethod
    def y_one_hot_encode_by_alphabet(class_index, n_class):
        y_input = np.zeros([n_class])
        y_input[int(class_index) - 1] = 1
        return y_input

    @staticmethod
    def load_csv(file_path, delimiter=',', quotechar='"'):
        data = []
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter, quotechar=quotechar)
            for row in reader:
                txt = ""
                for char in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", char).replace("\\n", "\n")
                data.append((int(row[0]), txt))
        return data

    def get_batch(self, text_list, class_list, length0, n_class):
        x_batch = np.expand_dims(
            np.concatenate(
                tuple([np.expand_dims(self.x_one_hot_encode_by_alphabet(text, length0), axis=0) for text in text_list])
            ),
            axis=-1
        )
        y_batch = np.concatenate(
            tuple([np.expand_dims(
                self.y_one_hot_encode_by_alphabet(class_index, n_class), axis=0) for class_index in class_list])
        )
        return x_batch, y_batch

    def get_batch_from_file(self, file_path, batch_size, length0, n_class):
        assert isinstance(file_path, str)
        text_list = []
        class_list = []
        if os.path.isfile(file_path):
            assert len(text_list) == len(class_list)
            # logging.info(config.START_TRAIN_FILE_MESS % file_path)
            # print(config.START_TRAIN_FILE_MESS % file_path)
            if os.path.splitext(file_path)[1] == '.csv':
                data = self.load_csv(file_path)
                while len(data) > 0 and len(text_list) < batch_size:
                    sample = data.pop(int(np.random.choice(len(data), 1)))
                    text_list.append(sample[1])
                    class_list.append(sample[0])
                    if len(text_list) == batch_size:
                        yield self.get_batch(text_list, class_list, length0, n_class)
                        text_list.clear()
                        class_list.clear()
        elif os.path.isdir(file_path):
            # logging.info(config.OPEN_DIR_MESS % file_path)
            # print(config.OPEN_DIR_MESS % file_path)
            for path in os.listdir(file_path):
                yield from self.get_batch_from_file(os.path.join(file_path, path), batch_size, length0, n_class)


if __name__ == "__main__":
    test_preparation = PreparationForTraining()
    for x_batch_test, y_batch_test in test_preparation.get_batch_from_file(config.TEST_TEXT_HOME, 512, 1014, 4):
        print(x_batch_test.shape, y_batch_test.shape)
