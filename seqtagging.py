# -*- coding: utf-8 -*-

# effort of writing python 2/3 compatiable code
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future.utils import iteritems
from operator import itemgetter, attrgetter, methodcaller

# from sys import stdin
# reload(sys)
# sys.setdefaultencoding('utf8')

import sys

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

from statesearch import *

# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.

class SeqState:
    O_state = -1
    non_I_state_list = []

    def __init__(self,sentence,vt):
        #print(sentence, vt)
        self.action_history = []
        self.words = sentence
        self.vt =vt
        self.n_tags = vt.size()
        self.tag_idx = 0

        if SeqState.O_state == -1:
            SeqState.O_state = vt.w2i["O"]
            for i in range(vt.size()):
                if "I-" not in vt.i2w[i]:
                    SeqState.non_I_state_list.append(i)
            print("non I states", SeqState.non_I_state_list)
            print ("ntags",vt.size())


    # def get_action_set(self):
    #     return range(self.n_tags)

    def get_action_set(self):
        if (self.tag_idx > 0 and self.action_history[-1] == SeqState.O_state):
            return SeqState.non_I_state_list
        return range(self.n_tags)

    def get_action_set_withans(self,gold_ans):
        gold_act = gold_ans[self.tag_idx]
        return [[gold_act],[1]]
        #return self.get_action_set() # no trick is done here

    def is_end(self):
        assert len(self.action_history) <= len(self.words)
        if len(self.action_history) == len(self.words):
            return True
        else:
            return False

    def reward(self,golds):
        good = 0.0
        bad = 0.0

        for w, gold_tag, pred_tag in zip(self.words, golds, self.action_history):
            if gold_tag == pred_tag:
                good += 1
            else:
                bad += 1
        #print (good)
        #return good
        return good/len(golds)

    # the estimated final reward value of by treating the partial pass as the full actions, after executing the given partial actions;
    # if you do not know how to estimate the reward given the partial sequence, should return 0 here
    def estimated_reward(self, gold_ans, action):
        good = 0.9
        bad = 0.9
        actions = self.action_history[:] + [action]
        for i in range(len(actions)):
            if gold_ans[i] == actions[i]:
                good += 1
            else:
                bad += 1

        return good/len(actions)


class SeqTaggingModel():

    def __init__(self,init_grad,n_words,n_tags):
        self.model = dt.Model()
        self.learner = dt.SimpleSGDTrainer(self.model,e0=init_grad)
        self.E = self.model.add_lookup_parameters((n_words, 128))

        self.pH = self.model.add_parameters((32, 50*2))
        self.pO = self.model.add_parameters((n_tags, 32))

        self.builders=[
            dt.LSTMBuilder(1, 128, 50, self.model),
            dt.LSTMBuilder(1, 128, 50, self.model),
        ]


class SeqScoreExpressionState(SeqState):

    def __init__(self, nmodel,sentence,vw,vt,copy=False):
        #super(SeqScoreExpressionState, self).__init__(sentence,vt)
        #super(SeqScoreExpressionState, self).__init__()
        SeqState.__init__(self,sentence,vt)
        self.path_score_expression = 0
        self.score = 0
        self.nm = nmodel
        self.vw = vw

        if copy == False:
            UNK = self.vw.w2i["_UNK_"]
            sent = self.words
            f_init, b_init = [b.initial_state() for b in self.nm.builders]
            wembs = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in sent]
            self.fw = [x.output() for x in f_init.add_inputs(wembs)]
            self.bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]
            self.bw.reverse()
            self.H = dt.parameter(self.nm.pH)
            self.O = dt.parameter(self.nm.pO)

    def get_next_score_expressions(self, action_list):
        #assert len(action_list) == self.n_tags
        i_repr = dt.concatenate([self.fw[self.tag_idx],self.bw[self.tag_idx]])
        r_t = self.O*(dt.tanh(self.H * i_repr)) #results for all list
        res_list=[]
        for action in action_list:
            res_list.append(r_t[action])

        return [dt.concatenate(res_list), [0] * len(action_list)] #row or cols?

    def get_new_state_after_action(self, action,meta_info):
        assert action in self.get_action_set()
        new_state = self.clone()

        #will make it call helper function from parnet instead
        new_state.action_history.append(action)
        new_state.tag_idx +=1
        return new_state

    def clone(self):
        res = SeqScoreExpressionState(self.nm,self.words,self.vw,self.vt,copy=True)

        #will make it call helper function from parnet instead
        res.action_history = self.action_history[:]
        res.tag_idx = self.tag_idx

        res.fw = self.fw
        res.bw = self.bw
        res.H = self.H
        res.O = self.O

        return res

    def __str__(self):
        #return "<"+" ".join(self.words) + "> at idx:  " + str(self.tag_idx)
        return ">  " + str(self.tag_idx) + ": " + " ".join([str(x) for x in self.action_history])

def main():
    train_file="./data/conll-4types-bio-dev.txt"
    test_file="./data/conll-4types-bio-test.txt"

    train=list(util.read(train_file))
    test=list(util.read(test_file))
    # train = train[:1]
    # test = train
    words=[]
    tags=[]

    wc=Counter()

    for s in train:
        for w,p in s:
            words.append(w)
            tags.append(p)
            wc[w]+=1


    words.append("_UNK_")
    #words=[w if wc[w] > 1 else "_UNK_" for w in words]
    #tags.append("_START_")

    for s in test:
        for w,p in s:
            words.append(w)

    vw = util.Vocab.from_corpus([words])
    vt = util.Vocab.from_corpus([tags])

    # for i in range(vt.size()):
    #     print("tagidx,tag",i,vt.i2w[i])
    nwords = vw.size()
    ntags  = vt.size()


    neural_model = SeqTaggingModel(0.01,nwords,ntags)
    sm = BeamSearchInferencer(neural_model,5)

    for ITER in xrange(1000):
        random.shuffle(train)
        loss = 0
        for i,s in enumerate(train,1):
            dt.renew_cg() #very important! to renew the cg
            words = [x[0] for x in s]
            tags = [x[1] for x in s]
            tags_idxes = [vt.w2i[t] for t in tags]
            init_state = SeqScoreExpressionState(neural_model,words,vw,vt,False)
            loss += sm.beam_train_max_margin_with_answer_guidence(init_state,tags_idxes)

            #loss += sm.beam_train_expected_reward(init_state,tags_idxes)
            #loss += sm.beam_train_max_margin(init_state,tags_idxes)
            #loss += sm.beam_train_max_margin_with_goldactions(init_state,tags_idxes)
            #loss += sm.greedy_train_max_sumlogllh(init_state,tags_idxes)

            if i % 1000 == 0:
                print (i)

        neural_model.learner.update_epoch(1.0)

        accuracy = 0.0
        total = 0.0
        oidx = vt.w2i['O']
        ocount = 0.0
        for i,s in enumerate(test,1):
            dt.renew_cg() #very important! to renew the cg
            words = [x[0] for x in s]
            tags = [x[1] for x in s]
            tags_idxes = [vt.w2i[t] for t in tags]
            init_state = SeqScoreExpressionState(neural_model,words,vw,vt,False)
            top_state = sm.beam_predict(init_state)[0]
            #top_state = sm.greedy_predict(init_state)
            for t in top_state.action_history:
                if t == oidx:
                    ocount += 1

            accuracy += top_state.reward(tags_idxes)

            total += len(tags)

        print ("accuracy",accuracy/total)
        print ("O percentage",ocount/total)

        print("In epoch ", ITER, " avg loss (or negative reward) is ", loss)

if __name__ == '__main__':
    main()
