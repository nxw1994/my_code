#!/usr/bin/env python
#coding=utf8

import math
import numpy as np
import logging

logging.basicConfig(format='[%(levelname)s]  %(message)s  '
                           '[%(asctime)s  %(filename)s  '
                           '(line:%(lineno)d)]',
                    level=logging.DEBUG)


def load_raw_texts(fname, valid_labels=None, with_label=True):
    texts = []
    labels = []
    with open(fname) as fin:
        i = 0
        for line in fin:
            line = line.strip().decode('utf8')
            i += 1
            if i % 100000 == 0:
                logging.info('finished %d lines' % i)
            if with_label:
                parts = line.split(' ', 1)
                assert (len(parts) == 2)
                if valid_labels and parts[0] in valid_labels:
                    labels.append(parts[0])
                    texts.append(parts[1])
            else:
                texts.append(line)

    return texts, labels


def data_generator(X, y, batch_size, shuffle=True):
    batch_num = int(math.ceil(y.shape[0] / float(batch_size)))
    counter = 0
    indices = np.arange(y.shape[0])
    if shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)

    while counter < batch_num:
        begin = batch_size * counter
        end = batch_size * (counter + 1)
        batch_indices = indices[begin:end]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        yield X_batch, y_batch

        counter += 1


