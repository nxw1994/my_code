#!/usr/bin/env python
# coding=utf8

import logging
import numpy as np
import os
import sys
import time
import tensorflow as tf
from sklearn.metrics import classification_report

from data_utils import data_generator
from dictionary import Dictionary
from tf_models import HAN

logging.basicConfig(format='[%(levelname)s]  %(message)s  '
                           '[%(asctime)s  %(filename)s  '
                           '(line:%(lineno)d)]',
                    level=logging.DEBUG)

reload(sys)
sys.setdefaultencoding('utf8')

# Constants {{{.
TRAIN_FILE = './data/train'
#VALID_FILE = None
#TEST_FILE = None
VALID_FILE = './data/dev'
TEST_FILE = './data/test'

EMBED_FILE = './data/embedding/39.train_valid.words.w2v.128.txt'
VOCAB_FILE = './data/embedding/dictionary1.txt'

checkpoint_dir = './out/'

EPOCH_NUM = 10
NUM_CHECKPOINTS = 3
VALID_RATIO = 0.2
MAX_SEQUENCE_LENGTH = 500

LEARNING_RATE = 0.0005
KEEP_PROB = 0.5
MAX_SENT_IN_DOC = 15
MAX_WORD_IN_SENT = 20
EMBEDDING_DIM = 128
BATCH_SIZE = 128

log_dir = './out/log_batch_{}_lr_{}_keep_prob_{}.txt'.format(BATCH_SIZE, LEARNING_RATE, KEEP_PROB)
model_name = 'han_batch_{}_lr_{}_keep_prob_{}'.format(BATCH_SIZE, LEARNING_RATE, KEEP_PROB)

VALID_LABELS = [
    'abroad', 'agriculture', 'astro', 'auto', 'baby', 'beauty',
    'career', 'comic', 'creativity', 'cul', 'digital', 'edu',
    'emotion', 'ent', 'finance', 'food', 'funny', 'game',
    'health', 'history', 'house', 'houseliving', 'inspiration',
    'law', 'life', 'lifestyle', 'lottery', 'mil', 'pet',
    'photography', 'politics', 'religion', 'science', 'social',
    'sports', 'tech', 'travel', 'weather', 'women']
# }}}.

# Global variables {{{.
cat_label_obj = None
# }}}.

log = open(log_dir, 'w')

class CatLabel(object):
    def __init__(self, text_labels):
        self.label2id = {}
        self.id2label = {}
        for l in set(text_labels):
            idx = len(self.label2id)
            self.label2id[l] = len(self.label2id)
            self.id2label[idx] = l

    def label_to_id(self, labels):
        return np.asarray([self.label2id[l] for l in labels])

    def id_to_label(self, ids):
        return np.asarray([self.id2label[i] for i in ids])

    @property
    def class_num(self):
        return len(self.label2id)


def load_embedding(fname, word_index):
    embedding_matrix = np.random.random((len(word_index), EMBEDDING_DIM))
    with open(fname) as fin:
        for i, line in enumerate(fin):
            values = line.decode('utf8').split(' ')
            word = values[0]
            if word not in word_index:
                continue

            vec = np.asarray(values[1:], dtype='float32')
            idx = word_index[word]
            embedding_matrix[idx] = vec

    return embedding_matrix


def load_raw_text_data(fname, valid_labels=None, with_label=True):
    texts = []
    labels = []
    with open(fname) as fin:
        for line in fin:
            line = line.strip().decode('utf8')
            if with_label:
                parts = line.split('\t', 1)
                assert (len(parts) == 2)
                if valid_labels and parts[0] in valid_labels:
                    labels.append(parts[0])
                    texts.append(parts[1])
            else:
                texts.append(line)

    return texts, labels


def prepare_data():
    logging.info('Loading train corpus...')
    sys.stdout.flush()
    train_texts, train_labels = load_raw_text_data(TRAIN_FILE, VALID_LABELS, True)
    logging.info('Finished.')
    sys.stdout.flush()

    # Prepare train data.
    logging.info('Transforming train data...')
    train_X = dict_obj.texts_to_sequences_han(train_texts, MAX_SENT_IN_DOC, MAX_WORD_IN_SENT)
    train_texts = None
    train_y = cat_label_obj.label_to_id(train_labels)
    logging.info('===> train_X shape:{}'.format(np.shape(train_X)))
    logging.info('===> train_y shape:{}'.format(np.shape(train_y)))
    logging.info('Finished.')
    sys.stdout.flush()

    # Prepare validation data.
    if VALID_FILE:
        valid_texts, valid_labels = load_raw_text_data(VALID_FILE, VALID_LABELS, True)
        valid_X = dict_obj.texts_to_sequences_han(valid_texts, MAX_SENT_IN_DOC, MAX_WORD_IN_SENT)
        valid_texts = None
        valid_y = cat_label_obj.label_to_id(valid_labels)
    else:
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)
        X = train_X[indices]
        y = train_y[indices]
        valid_samples = int(VALID_RATIO * X.shape[0])
        train_X = X[:-valid_samples]
        train_y = y[:-valid_samples]
        valid_X = X[-valid_samples:]
        valid_y = y[-valid_samples:]

    # Prepare test data.
    test_X = None
    test_y = None
    if TEST_FILE:
        logging.info('Transforming test data...')
        test_texts, test_labels = load_raw_text_data(TEST_FILE, VALID_LABELS, True)
        test_X = dict_obj.texts_to_sequences_han(test_texts, MAX_SENT_IN_DOC, MAX_WORD_IN_SENT)
        test_texts = None
        test_y = cat_label_obj.label_to_id(test_labels)
        logging.info('Finished.')
        sys.stdout.flush()

    logging.info('===> train_X shape:{}'.format(np.shape(train_X)))
    logging.info('===> train_y shape:{}'.format(np.shape(train_y)))
    logging.info('===> valid_X shape:{}'.format(np.shape(valid_X)))
    logging.info('===> valid_y shape:{}'.format(np.shape(valid_y)))
    if TEST_FILE:
        logging.info('===> test_X shape:{}'.format(np.shape(test_X)))
        logging.info('===> test_y shape:{}'.format(np.shape(test_y)))

    return train_X, train_y, valid_X, valid_y, test_X, test_y


def train():
    train_X, train_y, valid_X, valid_y, test_X, test_y = prepare_data()

    # Load embedding {{{.
    '''
    logging.info('Init embedding...')
    sys.stdout.flush()
    init_embedding = load_embedding(EMBED_FILE, dict_obj.word_index)
    logging.info('Finished.')
    sys.stdout.flush()
    # }}}.
    '''

    # Create model {{{
    logging.info('Creating model...')
    sys.stdout.flush()

    # HAN
    han_model = HAN(vocab_size=len(dict_obj),
                    num_classes=len(VALID_LABELS),
                    learning_rate=LEARNING_RATE,
                    embedding_size=EMBEDDING_DIM)
    logging.info('Finished.')
    sys.stdout.flush()
    # }}}.

    # Train and evaluate {{{.
    logging.info('Training model...')
    with tf.Session() as sess:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        acc_total = []
        for epoch in range(EPOCH_NUM):
            logging.info('current epoch %s' % (epoch + 1))
            train_loss, train_acc = train_epoch(epoch, train_X, train_y, han_model, sess)
            dev_loss, dev_acc, _ = dev_epoch(valid_X, valid_y, han_model, sess)
            log.write('epoch: {}  train_loss: {} train_acc: {}  dev_loss: {}  dev_acc: {}\n'.
                      format(epoch+1, train_loss, train_acc, dev_loss, dev_acc))
            acc_total.append(float(dev_acc))

        log.write('Save model...\n')
        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=EPOCH_NUM)
        log.write('Model has been saved...model path is {}\n'.format(checkpoint_dir))
        logging.info('Evaluate model...')
        test_loss, test_acc, _ = dev_epoch(test_X, test_y, han_model, sess)
        log.write('Evaluate model: test loss: {}  test_acc: {}'.format(test_loss, test_acc))
    logging.info('Finished.')
    log.close()
    sys.stdout.flush()


def train_epoch(epoch, train_x, train_y, model, sess):
    losses = []
    accs = []
    batches = data_generator(train_x, train_y, BATCH_SIZE)
    for x_batch, y_batch in batches:
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.max_sentence_num: MAX_SENT_IN_DOC,
            model.max_sentence_length: MAX_WORD_IN_SENT,
            model.batch_size: BATCH_SIZE,
            model.keep_prob: KEEP_PROB
        }
        _, cost, accuracy = sess.run([model.train_op, model.loss, model.acc], feed_dict)
        logging.info(
            "++++++++++++++++++train++++++++++++++{}: step_loss {:g}, step_acc {:g}".format(epoch, cost, accuracy))
        losses += [cost]
        accs += [accuracy]
    ave_loss = sum(losses) / len(losses)
    ave_acc = sum(accs) / len(accs)
    logging.info(
        "ooooooooooooooooooooTRAINoooooooooooooooooo{}: ave_loss {:g}, ave_acc {:g}".format(epoch, ave_loss, ave_acc))
    return ave_loss, ave_acc


def dev_epoch(dev_x, dev_y, model, sess):
    losses = []
    accs = []
    predicts = []
    batches = data_generator(dev_x, dev_y, BATCH_SIZE)

    for x_batch, y_batch in batches:
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.max_sentence_num: MAX_SENT_IN_DOC,
            model.max_sentence_length: MAX_WORD_IN_SENT,
            model.batch_size: BATCH_SIZE,
            model.keep_prob: 1.0
        }
        cost, accuracy, predict = sess.run([model.loss, model.acc, model.predict], feed_dict)
        losses += [cost]
        accs += [accuracy]
        predicts.extend(predict)

    ave_loss = sum(losses) / len(losses)
    ave_acc = sum(accs) / len(accs)
    time_str = str(int(time.time()))
    logging.info(
        "ooooooooooooooooooooDEVoooooooooooooooooo{}: ave_loss {:g}, ave_acc {:g}".format(time_str, ave_loss, ave_acc))
    return ave_loss, ave_acc, predicts


def prepare_test_data(test_file):
    # Prepare test data.
    test_X = None
    test_y = None
    if test_file:
        logging.info('Transforming test data...')
        test_texts, test_labels = load_raw_text_data(test_file, VALID_LABELS, True)
        test_X = dict_obj.texts_to_sequences_han(test_texts, MAX_SENT_IN_DOC, MAX_WORD_IN_SENT)
        test_texts = None
        test_y = cat_label_obj.label_to_id(test_labels)
        logging.info('Finished.')
        sys.stdout.flush()

    logging.info('===> test_X shape:{}'.format(np.shape(test_X)))
    logging.info('===> test_y shape:{}'.format(np.shape(test_y)))
    return test_X, test_y


def test():
    test_X, test_y = prepare_test_data(TEST_FILE)
    
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint('../model_and_log/model7/')
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            max_sentence_num = graph.get_operation_by_name('placeholder/max_sentence_num').outputs[0]
            max_sentence_length = graph.get_operation_by_name('placeholder/max_sentence_length').outputs[0]
            input_x = graph.get_operation_by_name("placeholder/input_x").outputs[0]
            input_y = graph.get_operation_by_name("placeholder/input_y").outputs[0]
            keep_prob = graph.get_operation_by_name('placeholder/keep_prob').outputs[0]
            batch_size = graph.get_operation_by_name('placeholder/batch_size').outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("predict").outputs[0]

            acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

            # Generate batches for one epoch
            batches = data_generator(test_X, test_y, BATCH_SIZE)

            # Collect the predictions here
            all_predictions = []
            all_y_test = []
            for x_test_batch, y_test_batch in batches:
                batch_predictions = sess.run(predictions, {max_sentence_num: MAX_SENT_IN_DOC,
                                                           max_sentence_length: MAX_WORD_IN_SENT,
                                                           input_x: x_test_batch,
                                                           batch_size: BATCH_SIZE,
                                                           keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_y_test.extend(y_test_batch)

    # calculate accuracy
    correct_predictions = float(sum(all_predictions == all_y_test))

    logging.info("Total number of test examples: {}".format(len(all_y_test)))
    logging.info('+++++++++++++{}+++++++++++++'.format(correct_predictions))
    logging.info("Accuracy: {:g}".format(correct_predictions / float(len(all_y_test))))

    # transfer the id to label
    preds = cat_label_obj.id_to_label(all_predictions)
    targets = cat_label_obj.id_to_label(all_y_test)
    
    logging.info("++++++++++++++++++++++HAN MODEL+++++++++++++++++++")
    logging.info(classification_report(y_true=targets, y_pred=preds, digits=4))


def main():
    logging.info('Init dictionary object and cat_label object...')
    sys.stdout.flush()
    global dict_obj
    dict_obj = Dictionary(VOCAB_FILE, '<PAD>', '<UNK>')

    global cat_label_obj
    cat_label_obj = CatLabel(VALID_LABELS)
    logging.info('Finished.')
    sys.stdout.flush()

    #test()
    train()


if __name__ == '__main__':
    main()




