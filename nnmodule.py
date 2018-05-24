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

#----------------------------------------------------------------------

""" This class is used as a score combiner. """
class FeedForwardModel:
    def __init__(self, model, dim_input, dim_hidden=-1):
        if dim_hidden == -1:    # 
            dim_hidden = dim_input
        self.W1 = model.add_parameters((dim_hidden, dim_input))
        self.W2 = model.add_parameters((1, dim_hidden))

    """ This is used for initializing parameter expressions for each example. """
    def spawn_expression(self):
        return FeedForwardExp(self.W1, self.W2)

class FeedForwardExp:
    def __init__(self, W1, W2):
        self.W1 = dt.parameter(W1)
        self.W2 = dt.parameter(W2)

    """ input_exp should be a vector of the scores to be combined """
    def score_expression(self, input_exp):
        return self.W2 * dt.tanh(self.W1 * input_exp)

#----------------------------------------------------------------------

class QuestionColumnMatchModel:
    def __init__(self, model, dim_word_embedding):
        self.ColW = model.add_parameters((dim_word_embedding))
    def spawn_expression(self):
        return QuestionColumnMatchExp(self.ColW)

class QuestionColumnMatchExp:
    def __init__(self, ColW):
        self.ColW = dt.parameter(ColW)

    """ return a list of scores """
    def score_expression(self, qwVecs, qwAvgVec, qLSTMVec, colnameVec, colWdVecs):
        colPriorScore = dt.dot_product(self.ColW, colnameVec)
        colMaxScore = AvgMaxScore(qwVecs, colWdVecs)
        colAvgScore = AvgScore(qwAvgVec, colnameVec)
        colQLSTMScore = AvgScore(qLSTMVec, colnameVec)
        ret = [colPriorScore, colMaxScore, colAvgScore, colQLSTMScore]
        return ret

#----------------------------------------------------------------------

class NegationModel:
    def __init__(self, model, dim_word_embedding, UNK, init_keywords = '', vw = None, E = None):
        if init_keywords:
            vector = dt.average([E[vw.w2i.get(w, UNK)] for w in init_keywords.split(' ')])
            self.NegW = model.parameters_from_numpy(vector.npvalue())
        else:
            self.NegW = model.add_parameters((dim_word_embedding))
    def spawn_expression(self):
        return NegationExp(self.NegW)

class NegationExp:
    def __init__(self, NegW):
        self.NegW = dt.parameter(NegW)

    def score_expression(self, qwAvgVec):
        ret = dt.dot_product(qwAvgVec, self.NegW)
        return ret

#----------------------------------------------------------------------

class CompareModel:
    def __init__(self, model, dim_word_embedding, UNK, init_keywords = '', vw = None, E = None):
        if init_keywords:
            vector = dt.average([E[vw.w2i.get(w, UNK)] for w in init_keywords.split(' ')])
            self.OpW = model.parameters_from_numpy(vector.npvalue())
        else:
            self.OpW = model.add_parameters((dim_word_embedding))
    def spawn_expression(self):
        return CompareExp(self.OpW)

class CompareExp:
    def __init__(self, OpW):
        self.OpW = dt.parameter(OpW)

    def score_expression(self, qwVecs, numWdPos):
        if numWdPos == 0:
            kwVec = qwVecs[numWdPos+1]
        elif numWdPos == 1:
            kwVec = qwVecs[0]
        else:
            kwVec = dt.average(qwVecs[numWdPos-2:numWdPos])

        ret = dt.dot_product(kwVec, self.OpW)
        return ret

#----------------------------------------------------------------------

class ArgModel:
    def __init__(self, model, dim_word_embedding, UNK, init_keywords = '', vw = None, E = None):
        if init_keywords:
            vector = dt.average([E[vw.w2i.get(w, UNK)] for w in init_keywords.split(' ')])
            self.OpW = model.parameters_from_numpy(vector.npvalue())
        else:
            self.OpW = model.add_parameters((dim_word_embedding))
    def spawn_expression(self):
        return ArgExp(self.OpW)

class ArgExp:
    def __init__(self, OpW):
        self.OpW = dt.parameter(OpW)

    def score_expression(self, qwVecs):
        ret = MaxScore(qwVecs, self.OpW)
        return ret

#----------------------------------------------------------------------

def AvgVector(txtV):
    if type(txtV) == list:
        vec = dt.average(txtV)
    else:
        vec = txtV
    return vec

''' txtV1 and txtV2 can be either a vector or a list of vectors '''
def AvgScore(txtV1, txtV2):
    vec1 = AvgVector(txtV1)
    vec2 = AvgVector(txtV2)
    ret = dt.dot_product(vec1, vec2)
    return ret

''' both qwVecs and colWdVecs have to be lists of vectors '''
def AvgMaxScore(qwVecs, colWdVecs):
    ret = dt.average([dt.emax([dt.dot_product(qwVec, colWdVec) for qwVec in qwVecs]) for colWdVec in colWdVecs])
    return ret

def MaxScore(qwVecs, vec):
    ret = dt.emax([dt.dot_product(qwVec, vec) for qwVec in qwVecs])
    return ret

