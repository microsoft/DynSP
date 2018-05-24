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

from action import *
from statesearch import *
from Parse import *

######## START OF THE CODE ########

def main1():
    dat = util.get_labeled_questions(str("data/nt-13588_2.tsv"), "data")
    fLog = sys.stdout
    for i,qinfo in enumerate(dat,1):
        if qinfo.seq_qid[-1] != '0':
            parse = Parse()
            parse.type = Parse.FollowUp
            cond = Condition(3,Condition.OpEqRow,7)
            parse.conditions = [cond]
            pred = parse.run(qinfo,resinfo)

            fLog.write("(%s) %s\n" % (qinfo.seq_qid, qinfo.question))
            fLog.write("Answer: %s\n" % ", ".join(["(%d,%d)" % coord for coord in qinfo.answer_coordinates]))
            fLog.write("Predictions: %s\n" % ", ".join(["(%d,%d)" % coord for coord in pred]))
            fLog.write("\n")
            fLog.flush()

        # use the gold answers
        resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence,
                                    qinfo.answer_coordinates, qinfo.answer_column_idx)

def main():
    dat = util.get_labeled_questions(str("data/random-split-2-dev.tsv"), "data")
    
    for qinfo in dat:
        print("%s\t%s" % (qinfo.question, list(util.findNumbers(qinfo.ques_word_sequence))))

    print (len(dat))
    return

    reader = csv.DictReader(open("data/train.tsv", 'r'), delimiter=str('\t'))
    for row in reader:
        ques = row['question']
        #print("%s\t%s" % (ques, list(util.findNumbers(ques))))
        print("%s\t%s" % (ques, list(util.findNumbers(ques.lower().split(' ')))))

if __name__ == '__main__':
    main()
