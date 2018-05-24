# -*- coding: utf-8 -*-

# effort of writing python 2/3 compatiable code
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future.utils import iteritems
import sys,io,os

reload(sys)
sys.setdefaultencoding('utf8')

import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

from collections import Counter
import random
import time
import tempfile

import util
import marisa_trie

data_path = "./cache/"
d = {}

NaN=-9999999
OnlyOneBest = True
d["DropOut"] = True
d["partial_reward"] = True
d["ReduceRowCond"] = False

d["USE_PRETRAIN_WORD_EMBEDDING"] = True
d["WORD_EMBEDDING_DIM"] = 100

d["DIST_BIAS_DIM"] = 100

#d["USE_PRETRAIN_WORD_EMBEDDING"] = False
#d["WORD_EMBEDDING_DIM"] = 2

#d["record_path"] = data_path + "/glove.twitter.100d.trie"
d["LSTM_HIDDEN_DIM"] = 50
d["record_path"] = data_path + "/glove.6b.100d.trie"

# d["WORD_EMBEDDING_DIM"] = 50
# d["word_embeddings"] = 50
# d["record_path"] = data_path +"/senna.wiki.50d.trie"

d["recordtriestructure"] = "<" + "".join(["f"] * d["WORD_EMBEDDING_DIM"])
d["recordtriesep"] = u"|"
d["embeddingtrie"] = marisa_trie.RecordTrie(d["recordtriestructure"])
d["embeddingtrie"].mmap(d["record_path"])

d["beam_size"] = 15
d["NUM_ITER"] = 30

UPDATE_WORD_EMD = 0
NOUPDATE_WORD_EMD_ALL = 1
NOUPDATE_WORD_EMD_PRETRAIN = 2

d["updateEMB"] = UPDATE_WORD_EMD

#d["AnnotatedTableDir"] = "/home/scottyih/Neural-RL-SP/Arvind_data/annotated"
#d["dirTable"] = "file:///D:/ScottYih/Source/Repos/Neural-RL-SP/data"

d["AnnotatedTableDir"] = "Arvind_data/annotated"
d["dirTable"] = "data"

#d["guessLogPass"] = True
d["guessLogPass"] = False
#d["verbose-dump"] = True
d["verbose-dump"] = False

if __name__ == '__main__':
    print("test")
