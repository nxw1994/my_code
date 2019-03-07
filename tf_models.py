#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


class HAN:
    def __init__(self, vocab_size, num_classes, learning_rate=0.001, decay_rate=0.98, decay_step=100,
                 embedding_init=None, embedding_size=128, hidden_size=128, l2_reg_lambda=0.1):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_init = embedding_init
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.l2_loss = tf.constant(0.0)

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.gloabl_steps,
                                                   self.decay_step,
                                                   self.decay_rate,
                                                   staircase=True)

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32,  name='keep_prob')

        word_embedded = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        doc_vec = self.doc2vec(sent_vec)
        out = self.classifer(doc_vec)

        self.out = out
        self.predict = tf.argmax(self.out, axis=1, name='predict')

        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.out,
                                                                    name='loss')
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predict, self.input_y)
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='acc')

        with tf.variable_scope('optimizer'):
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_pad = tf.Variable(tf.constant(0.0, shape=[1, self.embedding_size], dtype=tf.float32),
                                        trainable=False)
            if self.embedding_init is None:
                embedding_other = tf.Variable(tf.truncated_normal((self.vocab_size - 1, self.embedding_size)),
                                              dtype=tf.float32)
            else:
                embedding_other = tf.Variable(self.embedding_init[1:], dtype=tf.float32)
            embedding_mat = tf.concat([embedding_pad, embedding_other], axis=0)
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            word_embedded_drop = tf.nn.dropout(word_embedded, keep_prob=self.keep_prob)
        return word_embedded_drop

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            W = tf.get_variable('W', [doc_vec.shape[-1], self.num_classes],
                                initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
            b = tf.Variable(tf.constant(0, shape=[self.num_classes],  dtype=tf.float32), name='b')
            # out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            out = tf.nn.xw_plus_b(doc_vec, W, b)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output


if __name__ == '__main__':
    a = HAN(50, 10)
    x = np.random.randint(0, 50, size=100)
    x = np.reshape(x, [4, 5, 5])
    y = [1, 0, 3, 2]

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10):
        feed_dict = {a.input_x: x,
                     a.input_y: y,
                     a.batch_size: 4,
                     a.max_sentence_length: 5,
                     a.max_sentence_num: 5,
                     a.keep_prob: 0.5}

        _, predict, acc, loss = sess.run([a.train_op, a.predict, a.acc, a.loss], feed_dict=feed_dict)
        print(predict, acc, loss)
