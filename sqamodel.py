# -*- coding: utf-8 -*-

# effort of writing python 2/3 compatiable code
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future.utils import iteritems
from operator import itemgetter, attrgetter, methodcaller

import sys, time, argparse, csv
import cProfile

if sys.version < '3':
    from codecs import getwriter
    stderr = getwriter('utf-8')(sys.stderr)
    stdout = getwriter('utf-8')(sys.stdout)
else:
    stderr = sys.stderr

import dynet as dt
from collections import Counter
import random
import util
import config
import cPickle
import copy

from action import *
from statesearch import *
import nnmodule as nnmod 

######## START OF THE CODE ########

class SqaModel():

    WORD_EMBEDDING_DIM = config.d["WORD_EMBEDDING_DIM"]
    LSTM_HIDDEN_DIM = config.d["LSTM_HIDDEN_DIM"]

    def __init__(self, init_learning_rate, vw, reload_embeddings = True):
        self.model = dt.Model()

        self.vw = vw

        UNK = self.vw.w2i["_UNK_"]
        n_words = vw.size()
        print("init vw =", self.vw.size(), "words")
        self.learning_rate = init_learning_rate
        #self.learner = dt.SimpleSGDTrainer(self.model, e0=init_learning_rate)
        self.learner = dt.SimpleSGDTrainer(self.model)
        self.E = self.model.add_lookup_parameters((n_words, SqaModel.WORD_EMBEDDING_DIM))
        # similarity(v,o): v^T o
        self.SelHW = self.model.add_parameters((4 * SqaModel.WORD_EMBEDDING_DIM))
        self.SelIntraFW = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM / 2, SqaModel.WORD_EMBEDDING_DIM))
        self.SelIntraHW = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM, SqaModel.WORD_EMBEDDING_DIM * 2))
        self.SelIntraBias = self.model.add_parameters((config.d["DIST_BIAS_DIM"]))
        self.ColTypeN = self.model.add_parameters((1))
        self.ColTypeW = self.model.add_parameters((1))
        self.NulW = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM))

        ''' new ways to add module '''
        self.SelColFF = nnmod.FeedForwardModel(self.model, 4)
        self.WhereColFF = nnmod.FeedForwardModel(self.model, 5)
        self.QCMatch = nnmod.QuestionColumnMatchModel(self.model, SqaModel.WORD_EMBEDDING_DIM)
        self.NegFF = nnmod.FeedForwardModel(self.model, 2)
        self.FpWhereColFF = nnmod.FeedForwardModel(self.model, 9)

        
        # LSTM question representation
        self.builders = [
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model),
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model)
        ]
        self.pH = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM*2))

        # LSTM question representation
        self.prev_builders = [
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model),
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model)
        ]
        self.prev_pH = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM*2))
        self.SelColFpWhereW = self.model.add_parameters((4))
        self.SameAsPreviousW = self.model.add_parameters((2))

        if config.d["USE_PRETRAIN_WORD_EMBEDDING"] and reload_embeddings:
            n_hit_pretrain = 0.0
            trie = config.d["embeddingtrie"]
            print ("beginning to load embeddings....")
            for i in range(n_words):
                word = self.vw.i2w[i].lower()
                results = trie.items(word+ config.d["recordtriesep"])
                if len(results) == 1:
                    pretrain_v = np.array(list(results[0][1]))
                    pretrain_v = pretrain_v/np.linalg.norm(pretrain_v)
                    self.E.init_row(i,pretrain_v)
                    n_hit_pretrain += 1
                else:
                    pretrain_v = self.E[i].npvalue()
                    pretrain_v = pretrain_v/np.linalg.norm(pretrain_v)
                    self.E.init_row(i,pretrain_v)
          

            print ("the number of words that are in pretrain", n_hit_pretrain, n_words, n_hit_pretrain/n_words)
            print ("loading complete!")

        if config.d["USE_PRETRAIN_WORD_EMBEDDING"]:
            self.Negate = nnmod.NegationModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "not", self.vw, self.E)
            self.CondGT = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "more greater larger than", self.vw, self.E)
            self.CondGE = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "more greater larger than or equal to at least", self.vw, self.E)
            self.CondLT = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "less fewer smaller than", self.vw, self.E)
            self.CondLE = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "less fewer smaller than or equal to at most", self.vw, self.E)
            self.ArgMin = nnmod.ArgModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "least fewest smallest lowest shortest oldest", self.vw, self.E)
            self.ArgMax = nnmod.ArgModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK, "most greatest biggest largest highest longest latest tallest", self.vw, self.E)

        else:
            self.Negate = nnmod.NegationModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.CondGT = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.CondGE = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.CondLT = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.CondLE = nnmod.CompareModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.ArgMin = nnmod.ArgModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
            self.ArgMax = nnmod.ArgModel(self.model, SqaModel.WORD_EMBEDDING_DIM, UNK)
    
    def save(self,header):
        print("Saving model with header = ", header)
        f = open(header + "-extra.bin",'wb')
        cPickle.dump(self.vw,f)
        cPickle.dump(self.learning_rate,f)
        f.close()
        self.model.save(header + "-dynetmodel.bin")
        #print("Done!")

    @staticmethod
    def load(header):
        print("Loading model with header = ", header)
        f = open(header + "-extra.bin",'rb')
        vw = cPickle.load(f)
        lr = cPickle.load(f)
        f.close()
        res = SqaModel(lr,vw,False) # do not waste time reload embeddings
        #res.model.load(header + "-dynetmodel.bin")
        res.model.populate(header + "-dynetmodel.bin")
        #print("Done!")

        return res
