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


class SqaState:
    action_factory_cache = {}   # qinfo.qid -> action_factory

    def __init__(self, qinfo, resinfo=None):
        if resinfo == None:
            if qinfo.seq_qid in SqaState.action_factory_cache:
                self.af = SqaState.action_factory_cache[qinfo.seq_qid]
            else:
                self.af = ActionFactory(qinfo)  # define the actions
                SqaState.action_factory_cache[qinfo.seq_qid] = self.af
        else:   # has previous question, currently no cache
            self.af = ActionFactory(qinfo, resinfo)  # define the actions
        self.action_history = [self.af.start_action_idx]    # fill in the "start" null action
        self.meta_history = [dt.inputVector(np.zeros(len(qinfo.ques_word_sequence)))]
        self.qinfo = qinfo
        self.resinfo = resinfo
        self.numCol = self.qinfo.num_columns
        self.numRow = self.qinfo.num_rows

    # Given the current state (self), return a list of legitimate actions
    # Currently, it follows the STAGG's fashion and requires it does SELECT first. We can later relax it.
    def get_action_set(self, action_history = None):
        if action_history == None:
            action_history = self.action_history
        last_act_idx = action_history[-1]
        return self.af.legit_next_action_idxs(last_act_idx, action_history)

    def is_end(self, action_history = None):
        return not self.get_action_set(action_history) # empty action_set

    # return a set of actions that can lead to the gold state from the current state
    def get_action_set_withans(self, gold_ans):
        ret_action = []
        ret_estimated_reward = []
        '''
        print("debug: current action_history", self.action_history)
        print("debug: self.get_action_set()", self.get_action_set())
        print("debug: self.qinfo.question", self.qinfo.question)
        for actidx in self.get_action_set():
            print("debug: action %d: %s" % (actidx, self.af.actidx2str(actidx)))
        print("")
        '''
        for act in self.get_action_set():
            e_reward = self.estimated_reward(gold_ans, act)
            #print("debug: act = %s, e_reward = %f" % (self.af.actidx2str(actidx), e_reward))
            if e_reward > 0:
                ret_action.append(act)
                ret_estimated_reward.append(e_reward)
        return ret_action, ret_estimated_reward

    # the estimated final reward value of a full path, after executing the given action
    def estimated_reward(self, gold_ans, act_idx):
        # if this action is the final action to goal state (i.e., Stop)
        # use the real reward directly
        if not self.af.legit_next_action_idxs(act_idx, self.action_history): # empty set
            path = self.action_history + [act_idx]
            ret = self.reward(gold_ans, path, True)
        else: 
            act_idx_stop = self.af.type2actidxs[ActionType.Stop][0]
            action = self.af.actions[act_idx]
            #print ("action.type:", action.type)
            if action.type == ActionType.Select:
                path = self.action_history + [act_idx, act_idx_stop]
                ret = self.reward(gold_ans, path, True)
            elif action.type == ActionType.WhereCol:    # TODO: ignore WhereCol although technically it's not correct
                path = self.action_history + [act_idx_stop]
                #print("path=", path)
                ret = self.reward(gold_ans, path, True)
            elif action.type == ActionType.FpWhereCol:    # ignore FpWhereCol & treat it as SameAsPrevious
                act_idx_same = self.af.type2actidxs[ActionType.SameAsPrevious][0]
                path = self.action_history + [act_idx_same]
                ret = self.reward(gold_ans, path, True)
            else: # append action
                path = self.action_history + [act_idx]
                #print("path2=", path, "path2.action.type=", [self.af.actions[p].type for p in path])
                ret = self.reward(gold_ans, path, True)

        if ret == 1: # check if we like this parse
           ret = self.af.action_history_quality(path)

        return ret

    def reward(self, gold, action_history = None, partial = None):
        if partial == None:
            partial = config.d["partial_reward"]

        if not gold:
            gold = self.qinfo.answer_coordinates

        # execute the parse
        pred = self.execute_parse(action_history)

        setGold = set(gold)
        setPred = set(pred)

        if partial:
            # Reward = #(Gold INTERSECT Pred) / #(Gold UNION Pred)
            ret = float(len(setGold.intersection(setPred))) / len(setGold.union(setPred))
        else:       
            # change the reward function to be 0/1
            if setGold == setPred:
                ret = 1.0
            else:
                ret = 0.0

        self.recent_pred = pred
        return ret

    def execute_parse(self, action_history=None):
        if action_history == None:
            action_history = self.action_history

        # only execute if the parse is complete
        if not self.is_end(action_history):
            return []

        # map the action sequence to a parse
        parse = self.af.action_history_to_parse(action_history)
        coords = parse.run(self.qinfo, self.resinfo)

        return coords

class SqaScoreExpressionState(SqaState):

    def __init__(self, nmodel, qinfo, init_example = True, resinfo = None, testmode = False):
        SqaState.__init__(self, qinfo, resinfo)
        self.path_score_expression = dt.scalarInput(0)
        self.score = 0
        self.nm = nmodel
        self.vw = self.nm.vw
        self.H = dt.parameter(self.nm.pH)
        self.prev_H = dt.parameter(self.nm.prev_pH)

        if init_example:
            UNK = self.vw.w2i["_UNK_"]

            # vectors of question words
            if testmode or not config.d["DropOut"]:
                self.ques_emb = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in self.qinfo.ques_word_sequence]
            else:
                self.ques_emb = [dt.dropout(self.nm.E[self.vw.w2i.get(w, UNK)], 0.5) for w in self.qinfo.ques_word_sequence]
            self.ques_avg_emb = dt.average(self.ques_emb)

            # column name embeddings
            self.colname_embs = []
            # avg. vectors of column names
            self.headers_embs = []
            for colname_word_sequence in self.qinfo.headers_word_sequences:
                colname_emb = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in colname_word_sequence]
                self.colname_embs.append(colname_emb)
                self.headers_embs.append(dt.average(colname_emb))

            # avg. vectors of table entries
            self.entries_embs = []
            for row_word_sequences in self.qinfo.entries_word_sequences:
                row_embs = []
                for cell_word_sequence in row_word_sequences:
                    row_embs.append(dt.average([self.nm.E[self.vw.w2i.get(w, UNK)] for w in cell_word_sequence]))
                self.entries_embs.append(row_embs)

            self.NulW = dt.parameter(self.nm.NulW)
            self.SelHW = dt.parameter(self.nm.SelHW)
            self.SelIntraFW = dt.parameter(self.nm.SelIntraFW)
            self.SelIntraHW = dt.parameter(self.nm.SelIntraHW)
            self.SelIntraBias = dt.parameter(self.nm.SelIntraBias)
            self.ColTypeN = dt.parameter(self.nm.ColTypeN)
            self.ColTypeW = dt.parameter(self.nm.ColTypeW)

            ''' new ways to add module '''
            self.SelColFF = self.nm.SelColFF.spawn_expression()
            self.WhereColFF = self.nm.WhereColFF.spawn_expression()
            self.QCMatch = self.nm.QCMatch.spawn_expression()
            self.Negate = self.nm.Negate.spawn_expression()
            self.NegFF = self.nm.NegFF.spawn_expression()
            self.FpWhereColFF = self.nm.FpWhereColFF.spawn_expression()
            self.CondGT = self.nm.CondGT.spawn_expression()
            self.CondGE = self.nm.CondGE.spawn_expression()
            self.CondLT = self.nm.CondLT.spawn_expression()
            self.CondLE = self.nm.CondLE.spawn_expression()
            self.ArgMin = self.nm.ArgMin.spawn_expression()
            self.ArgMax = self.nm.ArgMax.spawn_expression()
            
            # question LSTM
            f_init, b_init = [b.initial_state() for b in self.nm.builders]
            self.fw = [x.output() for x in f_init.add_inputs(self.ques_emb)]
            self.bw = [x.output() for x in b_init.add_inputs(reversed(self.ques_emb))]
            self.bw.reverse()

            # from previous question & its answers
            if resinfo != None:
                # vectors of question words
                self.prev_ques_emb = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in self.resinfo.prev_ques_word_sequence]
                self.prev_ques_avg_emb = dt.average(self.prev_ques_emb)

                # previous question LSTM
                f_init, b_init = [b.initial_state() for b in self.nm.prev_builders]
                self.prev_fw = [x.output() for x in f_init.add_inputs(self.prev_ques_emb)]
                self.prev_bw = [x.output() for x in b_init.add_inputs(reversed(self.prev_ques_emb))]
                self.prev_bw.reverse()

                self.SelColFpWhereW = dt.parameter(self.nm.SelColFpWhereW)
                self.SameAsPreviousW = dt.parameter(self.nm.SameAsPreviousW)

    def clone(self):
        res = SqaScoreExpressionState(self.nm, self.qinfo, False, self.resinfo)
        res.action_history = self.action_history[:]
        res.meta_history = self.meta_history[:]

        # vectors of question words
        res.ques_emb = self.ques_emb
        res.ques_avg_emb = self.ques_avg_emb

        # vectors of column names
        res.colname_embs = self.colname_embs
        res.headers_embs = self.headers_embs

        # avg. vectors of table entries
        res.entries_embs = self.entries_embs
        res.NulW = self.NulW

        res.SelHW = self.SelHW
        res.SelIntraFW = self.SelIntraFW
        res.SelIntraHW = self.SelIntraHW
        res.SelIntraBias = self.SelIntraBias
        res.ColTypeN = self.ColTypeN
        res.ColTypeW = self.ColTypeW
        res.fw = self.fw
        res.bw = self.bw

        ''' clone '''
        '''
        res.SelColFF = copy.deepcopy(self.SelColFF)
        res.WhereColFF = copy.deepcopy(self.WhereColFF)
        res.QCMatch = copy.deepcopy(self.QCMatch)
        res.Negate = copy.deepcopy(self.Negate)
        res.NegFF = copy.deepcopy(self.NegFF)
        res.FpWhereColFF = copy.deepcopy(self.FpWhereColFF)
        '''
        res.SelColFF = self.nm.SelColFF.spawn_expression()
        res.WhereColFF = self.nm.WhereColFF.spawn_expression()
        res.QCMatch = self.nm.QCMatch.spawn_expression()
        res.Negate = self.nm.Negate.spawn_expression()
        res.NegFF = self.nm.NegFF.spawn_expression()
        res.FpWhereColFF = self.nm.FpWhereColFF.spawn_expression()
        res.CondGT = self.nm.CondGT.spawn_expression()
        res.CondGE = self.nm.CondGE.spawn_expression()
        res.CondLT = self.nm.CondLT.spawn_expression()
        res.CondLE = self.nm.CondLE.spawn_expression()
        res.ArgMin = self.nm.ArgMin.spawn_expression()
        res.ArgMax = self.nm.ArgMax.spawn_expression()

        if self.resinfo != None:
            # vectors of previous question words
            res.prev_ques_emb = self.prev_ques_emb
            res.prev_ques_avg_emb = self.prev_ques_avg_emb

            # previous question LSTM
            res.prev_fw = self.prev_fw
            res.prev_bw = self.prev_bw

            res.SelColFpWhereW = self.SelColFpWhereW

        return res


    # Decomposable attention between question and column name
    # Overall, it needs more efficient implementation... :(
    def decomp_attend(self, vecsA, vecsB):
        # Fq^T Fc -> need to expedite using native matrix/tensor multiplication 
        Fq = vecsA     # the original word vector, not yet passing a NN as in Eq.1, # need a function F
        Fc = vecsB    # need a function F
                
        expE = []
        for fq in Fq:
            row = []
            for fc in Fc:
                row.append(dt.exp(dt.dot_product(fq,fc)))
            expE.append(row)
        #print ("debug: expE", expE[0][0].value())

        invSumExpEi = []
        for i in xrange(len(Fq)):
            invSumExpEi.append(dt.pow(dt.esum(expE[i]), dt.scalarInput(-1)))

        invSumExpEj = []
        for j in xrange(len(Fc)):
            invSumExpEj.append(dt.pow(dt.esum([expE[i][j] for i in xrange(len(Fq))]), dt.scalarInput(-1)))

        beta = []
        for i in xrange(len(Fq)):
            s = dt.esum([Fc[j] * expE[i][j] for j in xrange(len(Fc))])
            beta.append(s * invSumExpEi[i])
        #print("debug: beta", beta[0].value())

        alpha = []
        for j in xrange(len(Fc)):
            s = dt.esum([Fc[j] * expE[i][j] for i in xrange(len(Fq))])
            alpha.append(s * invSumExpEj[j])
        #print("debug: alpha", alpha[0].value())

        # Compare
        v1i = [dt.logistic(dt.concatenate([Fq[i],beta[i]])) for i in xrange(len(Fq))]       # need a function G
        v2j = [dt.logistic(dt.concatenate([Fc[j],alpha[j]])) for j in xrange(len(Fc))]      # need a function G

        #print ("debug: v1i", v1i[0].value())
        #print ("debug: v2j", v2j[0].value())

        # Aggregate
                
        v1 = dt.esum(v1i)
        v2 = dt.esum(v2j)

        #print ("debug: v1.value()", v1.value())
        #print ("debug: v2.value()", v2.value())

        #colScore = dt.logistic(dt.dot_product(self.SelHW, dt.concatenate([v1,v2])))
        return dt.dot_product(v1,v2)

    def intra_sent_attend(self, vecs):
        numVecs = len(vecs)
        fVecs = [dt.tanh(self.SelIntraFW * v) for v in vecs]
        expE = []
        for i,fq in enumerate(fVecs):
            row = []
            for j,fc in enumerate(fVecs):
                row.append(dt.exp(dt.dot_product(fq,fc) + self.SelIntraBias[i-j+int(config.d["DIST_BIAS_DIM"]/2)]))
            expE.append(row)

        invSumExpE = []
        for i in xrange(numVecs):
            invSumExpE.append(dt.pow(dt.esum(expE[i]), dt.scalarInput(-1)))

        alpha = []
        for i in xrange(numVecs):
            s = dt.esum([vecs[j] * expE[i][j] for j in xrange(numVecs)])
            alpha.append(s * invSumExpE[i])

        return [dt.tanh(self.SelIntraHW * dt.concatenate([v,a])) for v,a in zip(vecs, alpha)]

    def positional_reweight(self, vecs):
        return [v * dt.logistic(self.SelIntraBias[i]) for i,v in enumerate(vecs)]

    # input: question word vectors, averaged matched word vectors (e.g., column name or table entry)
    # output: a vector of length of question; each element represents how much the word is covered
    def determine_coverage_by_name(self, qwVecs, avgVec):
        return None
        # Compute question coverage -- hard/rough implementation to test idea first
        qWdMatchScore = [dt.dot_product(qwVec, avgVec).value() for qwVec in qwVecs]
        ret = dt.softmax(dt.inputVector(np.array(qWdMatchScore)))
        return ret
        
    def attend_question_coverage(self):
        return self.ques_emb
        #print("Question Info: seq_qid", self.qinfo.seq_qid, "question", self.qinfo.question)
        maskWdIndices = set()
        for coverageMap in self.meta_history:
            mask = coverageMap.value()
            if type(mask) != list:
                mask = [mask]
            max_value = max(mask)
            if max_value == 0:
                continue
            max_index = mask.index(max_value)
            maskWdIndices.add(max_index)
        qwVecs = []
        for i,vec in enumerate(self.ques_emb):
            #print (i, vec)
            if mask[i] == 1:
                qwVecs.append(dt.inputVector(np.zeros(SqaModel.WORD_EMBEDDING_DIM)))
            else:
                qwVecs.append(vec)
        return qwVecs


    def get_next_score_expressions(self, legit_act_idxs):

        res_list = []
        meta_list = []
        '''
        print ("debug: self.action_history", self.action_history)
        print ("debug: self.is_end()", self.is_end())
        print ("debug: self.qinfo.seq_qid", self.qinfo.seq_qid)
        print ("debug: legit_act_idxs", legit_act_idxs)
        '''

        qwVecs = self.attend_question_coverage()
        qwAvgVec = self.ques_avg_emb
        qLSTMVec = dt.tanh(self.H * dt.concatenate([self.fw[-1],self.bw[0]]))   # question words LSTM embedding

        if self.resinfo != None:
            prev_qwVecs = self.prev_ques_emb
            prev_qwAvgVec = self.prev_ques_avg_emb
            prev_qLSTMVec = dt.tanh(self.prev_H * dt.concatenate([self.prev_fw[-1], self.prev_bw[0]]))

        for act_idx in legit_act_idxs:
            action = self.af.actions[act_idx]
            act_type = action.type
            #print("act_type", act_type)

            col = action.col
            colnameVec = self.headers_embs[col]
            colWdVecs = self.colname_embs[col]
            r = action.row
            if self.action_history != []:
                # for condition check, assuming the last action of the current state is ActWhereCol
                c = self.af.actions[self.action_history[-1]].col
                condCellVec = self.entries_embs[r][c]

            if act_type == ActionType.Stop:
                # use the average after mask
                actScore = dt.dot_product(dt.average(qwVecs), self.NulW)
                coverageMap = dt.inputVector(np.zeros(len(qwVecs)))

            elif act_type == ActionType.Select:
                lstScores = self.QCMatch.score_expression(qwVecs, qwAvgVec, qLSTMVec, colnameVec, colWdVecs)
                scoreVec = dt.concatenate(lstScores)
                actScore = self.SelColFF.score_expression(scoreVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, colnameVec)

            elif act_type == ActionType.WhereCol:  # same as ActionType.ActSelect, but with different coefficients in weighted sum
                # column type embedding # TODO: MAY BE WRONG IMPLEMENTATION HERE
                if self.qinfo.values_in_ques:
                    colTypeScore = self.ColTypeN
                else:
                    colTypeScore = self.ColTypeW
                lstScores = self.QCMatch.score_expression(qwVecs, qwAvgVec, qLSTMVec, colnameVec, colWdVecs)
                scoreVec = dt.concatenate(lstScores + [colTypeScore])
                actScore = self.WhereColFF.score_expression(scoreVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, colnameVec)

            elif act_type == ActionType.CondEqRow:
                actScore = nnmod.MaxScore(qwVecs, condCellVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, condCellVec)

            elif act_type == ActionType.CondNeRow:
                entScore = nnmod.MaxScore(qwVecs, condCellVec)
                negScore = self.Negate.score_expression(qwAvgVec)
                scoreVec = dt.concatenate([entScore, negScore])
                actScore = self.NegFF.score_expression(scoreVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, condCellVec)

            elif act_type == ActionType.CondGT or act_type == ActionType.FpCondGT:
                actScore = self.CondGT.score_expression(qwVecs, action.val[0])
                coverageMap = self.determine_coverage_by_name(qwVecs, self.CondGT.OpW)

            elif act_type == ActionType.CondGE or act_type == ActionType.FpCondGE:
                actScore = self.CondGE.score_expression(qwVecs, action.val[0])
                coverageMap = self.determine_coverage_by_name(qwVecs, self.CondGE.OpW)

            elif act_type == ActionType.CondLT or act_type == ActionType.FpCondLT:
                actScore = self.CondLT.score_expression(qwVecs, action.val[0])
                coverageMap = self.determine_coverage_by_name(qwVecs, self.CondLT.OpW)

            elif act_type == ActionType.CondLE or act_type == ActionType.FpCondLE:
                actScore = self.CondLE.score_expression(qwVecs, action.val[0])
                coverageMap = self.determine_coverage_by_name(qwVecs, self.CondLE.OpW)

            elif act_type == ActionType.ArgMin or act_type == ActionType.FpArgMin:
                actScore = self.ArgMin.score_expression(qwVecs)
                coverageMap = self.determine_coverage_by_name(qwVecs, self.ArgMin.OpW)

            elif act_type == ActionType.ArgMax or act_type == ActionType.FpArgMax:
                actScore = self.ArgMax.score_expression(qwVecs)
                coverageMap = self.determine_coverage_by_name(qwVecs, self.ArgMax.OpW)

            elif act_type == ActionType.FpWhereCol:  # similar to ActionType.WhereCol
                # column type embedding
                if self.qinfo.values_in_ques:
                    colTypeScore = self.ColTypeN
                else:
                    colTypeScore = self.ColTypeW
                lstScores = self.QCMatch.score_expression(qwVecs, qwAvgVec, qLSTMVec, colnameVec, colWdVecs)
                lstPrevScores = self.QCMatch.score_expression(prev_qwVecs, prev_qwAvgVec, prev_qLSTMVec, colnameVec, colWdVecs)
                scoreVec = dt.concatenate(lstScores + [colTypeScore] + lstPrevScores)

                actScore = self.FpWhereColFF.score_expression(scoreVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, colnameVec)

            elif act_type == ActionType.FpCondEqRow:
                entScore = nnmod.MaxScore(qwVecs, condCellVec)
                prev_entScore = nnmod.MaxScore(prev_qwVecs, condCellVec)

                actScore = dt.bmax(entScore, prev_entScore)
                coverageMap = self.determine_coverage_by_name(qwVecs, condCellVec)

            elif act_type == ActionType.FpCondNeRow:
                entScore = nnmod.MaxScore(qwVecs, condCellVec)
                prev_entScore = nnmod.MaxScore(prev_qwVecs, condCellVec)
                negScore = self.Negate.score_expression(qwAvgVec)
                scoreVec = dt.concatenate([dt.bmax(entScore, prev_entScore), negScore])
                actScore = self.NegFF.score_expression(scoreVec)
                coverageMap = self.determine_coverage_by_name(qwVecs, condCellVec)


            elif act_type == ActionType.SameAsPrevious:
                quesLSTMScore = dt.dot_product(prev_qLSTMVec, qLSTMVec)
                quesAvgScore = dt.dot_product(prev_qwAvgVec, qwAvgVec)
                actScore = dt.dot_product(self.SameAsPreviousW,
                                           dt.concatenate([quesLSTMScore, quesAvgScore]))
                coverageMap = dt.inputVector(np.zeros(len(qwVecs)))

            else:
                assert False, "Error! Unknown act_type: %d" % act_type

            res_list.append(actScore)
            meta_list.append(coverageMap)

        return dt.concatenate(res_list), meta_list

    def get_new_state_after_action(self, action, meta):
        assert action in self.get_action_set()
        new_state = self.clone()
        new_state.action_history.append(action)
        new_state.meta_history.append(meta)
        return new_state

    def __str__(self):
        return "\t".join([self.af.actidx2str(act) for act in self.action_history])