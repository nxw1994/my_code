#!/usr/bin/env python
# coding=utf8

import logging
import numpy as np
import sys

logging.basicConfig(format='[%(levelname)s]  %(message)s  '
                           '[%(asctime)s  %(filename)s  '
                           '(line:%(lineno)d)]',
                    level=logging.DEBUG)


class Dictionary(object):
    def __init__(self, dict_file=None, pad_token=None, unk_token=None):
        self.pad = pad_token
        self.unk = unk_token
        self.word_index = {}
        self.index_word = {}
        self._load_dictionary(dict_file)

    def _load_dictionary(self, fname):
        def add_word(word):
            if word not in self.word_index:
                index = len(self.word_index)
                self.word_index[word] = index
                self.index_word[index] = word

        if self.pad and self.unk:
            add_word(self.pad)
            add_word(self.unk)

        with open(fname) as fin:
            for line in fin:
                line = line.strip().decode('utf8')
                add_word(line)
        logging.info('Finished loading dictionary, totally %d words' % len(self.word_index))
        sys.stdout.flush()

    def texts_to_sequences(self, texts):
        if not texts or not isinstance(texts, list):
            return
        seqs = []
        for text in texts:
            word_list = None
            if isinstance(text, str):
                word_list = text.split(' ')
            elif isinstance(text, list):
                word_list = text

            if word_list is None:
                continue

            seq = []
            for w in word_list:
                if w in self.word_index:
                    seq.append(self.word_index[w])
                elif self.unk:
                    seq.append(self.word_index[self.unk])

            seqs.append(seq)

        return np.asarray(seqs)

    def texts_to_sequences_han(self, texts, max_sent_in_doc, max_word_in_sent):
        if not texts or not isinstance(texts, list):
            return
        
        seqs = np.zeros([len(texts), max_sent_in_doc, max_word_in_sent])
        for k, text in enumerate(texts):
            sen_list = None
            
            #print 'OK1'
            sen_list = [s for s in text.encode('utf8').split('。') if s != ' ']

            if sen_list is None:
               #print 'OK2'
               continue

            doc = np.zeros([max_sent_in_doc, max_word_in_sent])
            for i, sen in enumerate(sen_list):
                if i < max_sent_in_doc:
                    word_to_index = np.zeros([max_word_in_sent], dtype=int)
                    for j, w in enumerate(sen.split()):
                        w = w.decode('utf8')
                        if j < max_word_in_sent:
                            if w in self.word_index:
                                word_to_index[j] = self.word_index[w]
                            elif self.unk:
                                word_to_index[j] = self.word_index[self.unk]
                    doc[i] = word_to_index
            seqs[k] = doc
        # print seqs
        return seqs

    def __len__(self):
        return len(self.word_index)


if __name__ == '__main__':
    dict_obj = Dictionary('./data/embedding/dictionary.txt', '<PAD>', '<UNK>')
    docs = dict_obj.texts_to_sequences_han([u'我 是 中国 人。我的家在东北。 。 。松花江 上 啊 啊 啊。第三方。 。阿水 放。东方 故事',
                                            u'今天 天气 。 。 很 好'],
                                           max_sent_in_doc=2,
                                           max_word_in_sent=4)
    print '--->', dict_obj.index_word[0]
    print docs
    print len(dict_obj)



