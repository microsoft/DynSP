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

import sys, time, argparse
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

from statesearch import *

######## START OF THE CODE ########

class SqaState:
    # Action type:
    # (1) SELECT X  (# table columns)
    # (2) WHERE NULL (no condition, 1)
    # (3) WHERE Y=? (# columns)
    # (4) WHERE Y=Z (# rows)
    # Legit sequence: (1) -> (2), (1) -> (3) -> (4)

    ActSelect, ActWhereNul, ActWhereCol, ActWhereEqRow = xrange(4)

    def __init__(self, qinfo):
        self.action_history = []
        self.qinfo = qinfo
        self.numCol = len(qinfo.headers)
        self.numRow = len(qinfo.entries)
        self.act2type = {}

        # Define the actions

        # ActSelect
        self.actSetSelectStartIdx = 0
        self.actSetSelect = xrange(self.actSetSelectStartIdx, self.actSetSelectStartIdx + self.numCol)
        for act in self.actSetSelect:
            self.act2type[act] = SqaState.ActSelect

        # ActWhereNul
        self.actSetWhereNulStartIdx = self.actSetSelectStartIdx + len(self.actSetSelect)
        self.actSetWhereNul = xrange(self.actSetWhereNulStartIdx, self.actSetWhereNulStartIdx + 1)
        for act in self.actSetWhereNul:
            self.act2type[act] = SqaState.ActWhereNul

        # ActWhereCol
        self.actSetWhereColStartIdx = self.actSetWhereNulStartIdx + len(self.actSetWhereNul)
        self.actSetWhereCol = xrange(self.actSetWhereColStartIdx, self.actSetWhereColStartIdx + self.numCol)
        for act in self.actSetWhereCol:
            self.act2type[act] = SqaState.ActWhereCol

        # ActWhereEqRow
        self.actSetWhereEqRowStartIdx = self.actSetWhereColStartIdx + len(self.actSetWhereCol)
        self.actSetWhereEqRow = xrange(self.actSetWhereEqRowStartIdx, self.actSetWhereEqRowStartIdx + self.numRow)
        for act in self.actSetWhereEqRow:
            self.act2type[act] = SqaState.ActWhereEqRow


    ### Auxiliary routines for action index mapping
    # Given the id of an action that belongs to ActSelect, return the column number
    def selectAct2Col(self, act):
        # check if actIdx belongs to the right action type
        if (self.act2type[act] != SqaState.ActSelect):
            return None
        col = act - self.actSetSelectStartIdx
        return col

    ### Auxiliary routines for action index mapping
    # Given the id of an action that belongs to ActWhereEq, return the table entry coordinate
    def whereEqAct2Coord(self, act):
        # check if actIdx belongs to the right action type
        if (self.act2type[act] != SqaState.ActWhereEq):
            return None
        idx = act - self.actSetWhereEqStartIdx
        r = idx // self.numCol
        c = idx % self.numCol
        return (r,c)

    ### Auxiliary routines for action index mapping
    # Given the id of an action that belongs to ActWhereCol, return the column number
    def whereColAct2Col(self, act):
        if (self.act2type[act] != SqaState.ActWhereCol):
            return None
        col = act - self.actSetWhereColStartIdx
        return col

    ### Auxiliary routines for action index mapping
    # Given the id of an action that belongs to ActWhereEqRow, return the row number
    def whereEqRowAct2Row(self, act):
        if (self.act2type[act] != SqaState.ActWhereEqRow):
            return None
        row = act - self.actSetWhereEqRowStartIdx
        return row

    ### Auxiliary routines for action index mapping
    # Given the coordinate of a table entry, return the ActWhereEq id
    # Not sure if needed now
    def coord2whereEqAct(self,r,c):
        return r*numCol + c + self.actSetWhereEqStartIdx


    # Given the current state (self), return a list of legitimate actions
    # Currently, it follows the STAGG's fashion and requires it does SELECT first. We can later relax it.
    #
    # (1) ActSelect: SELECT X  (# table columns)
    # (2) ActWhereNul: WHERE NULL (no condition, 1)
    # (3) ActWhereCol: WHERE Y=? (# columns)
    # (4) ActWhereEqRow: WHERE Y=Z (# rows)
    # Legit sequence: (1) -> (2), (1) -> (3) -> (4)
    #
    def get_action_set(self):
        if not self.action_history: # empty action_history
            return self.actSetSelect
        else:
            last_act = self.action_history[-1]
            if self.act2type[last_act] == SqaState.ActSelect:
                return list(self.actSetWhereNul) + list(self.actSetWhereCol)
            elif self.act2type[last_act] == SqaState.ActWhereCol:
                return list(self.actSetWhereEqRow)
        return []

    def is_end(self):
        #return len(self.action_history) == 1  # only SELECT X
        return not self.get_action_set() # empty action_set

    # return a set of action that can lead to the gold state from the current state
    def get_action_set_withans(self, gold_ans):
        ret = []
        for act in self.get_action_set():
            if self.estimated_reward(gold_ans, act) > 0:    # TODO: fix redundant calling estimated_reward by the search code
                ret.append(act)
        return ret

    # the estimated final reward value of a full path, after executing the given action
    def estimated_reward(self, gold_ans, action):
        # if this action is the final action to goal state (i.e., (2) ActWhereNul or (4) ActWhereEqRow)
        # use the real reward directly
        if self.act2type[action] == SqaState.ActWhereNul or self.act2type[action] == SqaState.ActWhereEqRow:
            path = self.action_history + [action]
            return self.reward(gold_ans, path)
        else: # treat it as select column only
            if self.act2type[action] == SqaState.ActSelect:
                return self.reward(gold_ans, [action, self.actSetWhereNul[0]])
            else: # action is (3) ActWhereCol:
                return self.reward(gold_ans, self.action_history + [self.actSetWhereNul[0]])

    # Reward = #(Gold INTERSECT Pred) / #(Gold UNION Pred)
    def reward(self, gold, action_history = None):
        if not gold:
            gold = self.qinfo.answer_coordinates

        # execute the parse
        pred = self.execute_parse(action_history)

        # if verbose:
        #     print("gold coordinates", gold)
        #     print("pred coordinates", pred)
        setGold = set(gold)
        setPred = set(pred)

        ret = float(len(setGold.intersection(setPred))) / len(setGold.union(setPred))
        #if (ret > 0 and ret < 1):
        #    print("qid", self.qinfo.seq_qid, "gold:", gold, "pred:", pred, "reward:", ret)

        return ret
        #return int(float(len(setGold.intersection(setPred))) / len(setGold.union(setPred))) #0-1 reward

    # Currently, it follows the STAGG's fashion and requires it does SELECT first. We can later relax it.
    #
    # (1) ActSelect: SELECT X  (# table columns)
    # (2) ActWhereNul: WHERE NULL (no condition, 1)
    # (3) ActWhereCol: WHERE Y=? (# columns)
    # (4) ActWhereEqRow: WHERE Y=Z (# rows)
    # Legit sequence: (1) -> (2), (1) -> (3) -> (4)
    #
    def execute_parse(self, action_history=None):
        if not action_history:
            action_history = self.action_history

        # only execute if the parse is complete (i.e., length-2 or length-3)
        #if len(action_history) != 1 and len(action_history) != 2: # and len(action_history) != 3:
        
        if len(action_history) != 2 and len(action_history) != 3:
            return []

        # answer column
        actSel = action_history[0]
        ans_col = self.selectAct2Col(actSel)

        #return [(r,ans_col) for r in xrange(self.numRow) if (r,ans_col) not in self.qinfo.illegit_answer_coordinates]

        # check where condition
        actWhere = action_history[1]
        if self.act2type[actWhere] == SqaState.ActWhereNul:
            legit_rows = [r for r in xrange(self.numRow)]
        elif self.act2type[actWhere] == SqaState.ActWhereCol:
            cond_col = self.whereColAct2Col(actWhere)
            actWhereEqRow = action_history[2]
            cond_row = self.whereEqRowAct2Row(actWhereEqRow)
            cond_val = self.qinfo.entries[cond_row][cond_col]
            legit_rows = [r for r in xrange(self.numRow) if self.qinfo.entries[r][cond_col].lower() == cond_val.lower()]
        
        return [(r,ans_col) for r in legit_rows]

    # For debugging
    def act2str(self, act):
        if self.act2type[act] == SqaState.ActSelect:
            col = self.selectAct2Col(act)
            return "SELECT %s" % self.qinfo.headers[col]
        elif self.act2type[act] == SqaState.ActWhereEq:
            r,c = self.whereEqAct2Coord(act)
            return "WHERE %s = '%s'" % (self.qinfo.headers[c], self.qinfo.entries[r][c])
        else: # self.act2type[act] == SqaState.ActWhereNul:
            return "WHERE True"

class SqaModel():

    WORD_EMBEDDING_DIM = config.d["WORD_EMBEDDING_DIM"]
    LSTM_HIDDEN_DIM = config.d["LSTM_HIDDEN_DIM"] 

    def __init__(self, init_learning_rate, vw):
        self.model = dt.Model()
        self.vw = vw
        n_words = vw.size()

        self.learner = dt.SimpleSGDTrainer(self.model, e0=init_learning_rate)
        self.E = self.model.add_lookup_parameters((n_words, SqaModel.WORD_EMBEDDING_DIM))
        # similarity(v,o): v^T o
        self.SelColW = self.model.add_parameters((4))
        self.SelColWhereW = self.model.add_parameters((4))
        self.NulW = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM))
        self.ColW = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM))

        # LSTM question representation
        self.builders=[
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model),
            dt.LSTMBuilder(1, SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM, self.model)
        ]
        self.pH = self.model.add_parameters((SqaModel.WORD_EMBEDDING_DIM, SqaModel.LSTM_HIDDEN_DIM*2))

        if config.d["USE_PRETRAIN_WORD_EMBEDDING"]:
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




class SqaScoreExpressionState(SqaState):

    def __init__(self, nmodel, qinfo, vw, init_example = True):
        SqaState.__init__(self, qinfo)
        self.path_score_expression = dt.scalarInput(0)
        self.score = 0
        self.nm = nmodel
        self.vw = vw
        self.H = dt.parameter(self.nm.pH)

        if init_example:
            UNK = self.vw.w2i["_UNK_"]

            # vectors of question words
            self.ques_emb = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in self.qinfo.ques_word_sequence]
            #self.ques_avg_emb = dt.average(self.ques_emb)
            #self.ques_emb = dt.concatenate_cols([self.nm.E[self.vw.w2i.get(w, UNK)] for w in self.qinfo.ques_word_sequence])

            # avg. vectors of column names
            self.headers_embs = []
            for colname_word_sequence in self.qinfo.headers_word_sequences:
                colname_emb = dt.average([self.nm.E[self.vw.w2i.get(w, UNK)] for w in colname_word_sequence])
                self.headers_embs.append(colname_emb)

            # avg. vectors of table entries
            self.entries_embs = []
            for row_word_sequences in self.qinfo.entries_word_sequences:
                row_embs = []
                for cell_word_sequence in row_word_sequences:
                    row_embs.append(dt.average([self.nm.E[self.vw.w2i.get(w, UNK)] for w in cell_word_sequence]))
                self.entries_embs.append(row_embs)

            self.NulW = dt.parameter(self.nm.NulW)
            self.ColW = dt.parameter(self.nm.ColW)
            self.SelColW = dt.parameter(self.nm.SelColW)
            self.SelColWhereW = dt.parameter(self.nm.SelColWhereW)

            # question LSTM
            f_init, b_init = [b.initial_state() for b in self.nm.builders]
            wembs = [self.nm.E[self.vw.w2i.get(w, UNK)] for w in self.qinfo.ques_word_sequence]
            self.fw = [x.output() for x in f_init.add_inputs(wembs)]
            self.bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]
            self.bw.reverse()


    def get_next_score_expressions(self, legit_actions):

        res_list = []
        for act in legit_actions:
            act_type = self.act2type[act]
            qwVecs = self.ques_emb
            qwAvgVec = dt.average(qwVecs)

            i_repr = dt.concatenate([self.fw[-1],self.bw[0]])
            qLSTMVec = dt.tanh(self.H * i_repr) # question words LSTM embedding

            if act_type == SqaState.ActSelect:
                # question_embedding x column_name_embedding
                col = self.selectAct2Col(act)
                colnameVec = self.headers_embs[col]

                colPriorScore = dt.dot_product(self.ColW, colnameVec)
                colMaxScore = dt.emax([dt.dot_product(qwVec, colnameVec) for qwVec in qwVecs])
                colAvgScore = dt.dot_product(qwAvgVec, colnameVec)
                colQLSTMScore = dt.dot_product(qLSTMVec, colnameVec)

                colScore = dt.dot_product(self.SelColW, dt.concatenate([colPriorScore, colMaxScore, colAvgScore, colQLSTMScore]))

                res_list.append(colScore)

            elif act_type == SqaState.ActWhereCol:  # same as SqaState.ActSelect
                # question_embedding x column_name_embedding
                col = self.whereColAct2Col(act)
                colnameVec = self.headers_embs[col]

                colPriorScore = dt.dot_product(self.ColW, colnameVec)
                colMaxScore = dt.emax([dt.dot_product(qwVec, colnameVec) for qwVec in qwVecs])
                colAvgScore = dt.dot_product(qwAvgVec, colnameVec)
                colQLSTMScore = dt.dot_product(qLSTMVec, colnameVec)

                colScore = dt.dot_product(self.SelColWhereW, dt.concatenate([colPriorScore, colMaxScore, colAvgScore, colQLSTMScore]))

                res_list.append(colScore)

            elif act_type == SqaState.ActWhereEqRow:
                r = self.whereEqRowAct2Row(act)
                c = self.whereColAct2Col(self.action_history[-1])   # assuming the last action of the curren state is ActWhereCol
                entryVec = self.entries_embs[r][c]
                # max_w sim(w,entry)
                entScore = dt.emax([dt.dot_product(qwVec, entryVec) for qwVec in qwVecs])
                res_list.append(entScore)

            elif act_type == SqaState.ActWhereNul:
                res_list.append(dt.dot_product(dt.average(qwVecs), self.NulW))

        return dt.concatenate(res_list)

    def get_new_state_after_action(self, action):
        assert action in self.get_action_set()
        new_state = self.clone()
        new_state.action_history.append(action)
        return new_state

    def clone(self):
        res = SqaScoreExpressionState(self.nm, self.qinfo, self.vw, False)
        res.action_history = self.action_history[:]

        # vectors of question words
        res.ques_emb = self.ques_emb
        #res.ques_avg_emb = self.ques_avg_emb

        # avg. vectors of column names
        res.headers_embs = self.headers_embs 
        
        # avg. vectors of table entries
        res.entries_embs = self.entries_embs
        res.ColW = self.ColW
        res.NulW = self.NulW

        res.SelColW = self.SelColW
        res.SelColWhereW = self.SelColWhereW
        res.fw = self.fw
        res.bw = self.bw

        return res

    def __str__(self):
        return ">  " + "\t".join([self.act2str(act) for act in self.action_history])

def main():
    
    parser = argparse.ArgumentParser(description='Targeting "first questions" only.')
    parser.add_argument('--expSym', help='1, 2, 3, 4, 5 or 0 (full)', type=int)
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-seed')
    args = parser.parse_args()

    # Prepare training and testing (development) data
    random.seed(1)

    data_folder = "data"
    if args.expSym == 0:
        print("Full Train/Test splits...")
        train_file="%s/train.first.tsv" % data_folder
        test_file="%s/test.first.tsv" % data_folder
    elif args.expSym in xrange(1,6):
        print ("Random-split-%d-train/dev..." % args.expSym)
        train_file="%s/random-split-%d-train.first.tsv" % (data_folder, args.expSym)
        test_file="%s/random-split-%d-dev.first.tsv" % (data_folder, args.expSym)
    else:
        print("Unknown experimental setting...")
        return

    print("=" * 80)
    print("Train",train_file)
    print("Test",test_file)
    print(config.d)
    print(">" * 8, "begin experiments")

    train = util.get_labeled_questions(train_file, data_folder)
    test = util.get_labeled_questions(test_file, data_folder)

    # create a word embedding table

    words = set(["_UNK_", "_EMPTY_"])
    for ex in train:
        words.update(ex.all_words)
    for ex in test:
        words.update(ex.all_words)

    vw = util.Vocab.from_corpus([words])
    nwords = vw.size()

    neural_model = SqaModel(0.01, vw)
    sm = BeamSearchInferencer(neural_model,config.d["beam_size"])

    # main loop
    start_time = time.time() 
    max_reward_at_epoch = [0,0]
    for ITER in xrange(config.d["NUM_ITER"]):
        random.shuffle(train)
        loss = 0
        for i,qinfo in enumerate(train,1):
            dt.renew_cg() # very important! to renew the cg

            init_state = SqaScoreExpressionState(neural_model, qinfo ,vw)
            #loss += sm.beam_train_max_margin(init_state, qinfo.answer_coordinates)
            loss += sm.beam_train_max_margin_with_answer_guidence(init_state, qinfo.answer_coordinates)

            if i % 100 == 0:
                print (i, "/", len(train))

        neural_model.learner.update_epoch(1.0)

        accuracy = 0.0
        all_reward = 0.0
        total = 0.0
        for i,qinfo in enumerate(test,1):
            dt.renew_cg() # very important! to renew the cg
            init_state = SqaScoreExpressionState(neural_model, qinfo ,vw)
            top1_state = sm.beam_predict(init_state)[0]
            rew = top1_state.reward(qinfo.answer_coordinates)

            all_reward += rew
            accuracy += int(rew)    # 0-1, only get a point if all predictions are correct
            total += 1

        print("In epoch ", ITER, " avg loss (or negative reward) is ", loss)
        reported_reward = all_reward/total
        reported_accuracy = accuracy/total
        print ("reward", reported_reward)
        print ("accuracy", reported_accuracy)
        if (reported_reward > max_reward_at_epoch[0]):
            max_reward_at_epoch = (reported_reward, reported_accuracy, ITER)

        now_time = time.time()
        print ("Time taken in this epoch", now_time - start_time)
        start_time = now_time
        print("Best Reward: %f (Accuracy: %f) at epoch %d" % max_reward_at_epoch)
        print ()
        sys.stdout.flush()

if __name__ == '__main__':
    cProfile.run('main()')
