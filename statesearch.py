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
import numpy as np
import random

class BeamSearchInferencer:
    def __init__(self,nmodel,beam_size = 5):
        self.beam_size = beam_size
        self.neural_model = nmodel
        self.only_one_best = True

    def beam_predict(self,init_state):
        """ perform beam search to find the state with the best path score: maximize \sum f(S,a)

        :param state: state
        :returns: a list of the state (usually with size beam)
        :rtype: list(State)

        """
        init_state.path_score_expression = dt.scalarInput(0)
        init_state.score = 0
        self.cur_save_states = [init_state]

        while True:
            self.next_states = []
            contain_nonend_state = False

            for state in self.cur_save_states:
                cur_path_expression = state.path_score_expression
                cur_path_score = cur_path_expression.scalar_value()

                if state.is_end():
                    self.next_states.append(state) # keep end state for once so that they can compete with longer sequence
                    continue

                contain_nonend_state = True

                action_list = state.get_action_set()
                new_expression_list, meta_info_list = state.get_next_score_expressions(action_list)

                for i in range(len(action_list)):
                    action = action_list[i]
                    new_expression = new_expression_list[i] #dynet call for getting element
                    meta_info = meta_info_list[i]
                    new_state = state.get_new_state_after_action(action,meta_info)
                    new_state.path_score_expression = cur_path_expression + new_expression
                    new_state.score = state.score + new_expression.scalar_value()
                    #print("comparison",new_state.score,new_state.path_score_expression.value())
                    #assert abs(new_state.score - new_state.path_score_expression.value()) < 1e-5

                    self.next_states.append(new_state)

            self.next_states.sort(key=lambda x: -1 * x.score)
            # for x in self.next_states:
            #     print("===>",x,x.score)

            next_size = min([len(self.next_states),self.beam_size])

            if contain_nonend_state == False:
                return self.next_states[:next_size]
            else:
                self.cur_save_states = self.next_states[:next_size]

    def beam_predict_max_violation(self,init_state,gold_ans):
        """ perform beam search to find the state with the best path score: maximize \sum f(S,a) - R(a) using estimated reward

        :param state: state
        :param state: the gold answers (used to calcuate step_reward)
        :returns: a list of the state (usually with size beam)
        :rtype: list(State)

        """
        init_state.path_score_expression = dt.scalarInput(0)
        init_state.score = 0
        self.cur_save_states = [init_state]

        while True:
            self.next_states = []
            contain_nonend_state = False

            for state in self.cur_save_states:
                cur_path_expression = state.path_score_expression
                cur_path_score = cur_path_expression.scalar_value()

                if state.is_end():
                    self.next_states.append(state) # keep end state for once so that they can compete with longer sequence
                    continue

                contain_nonend_state = True

                action_list = state.get_action_set()
                estimated_reward_list = [state.estimated_reward(gold_ans,action) for action in action_list]


                new_expression_list, meta_info_list = state.get_next_score_expressions(action_list)


                for i in range(len(action_list)):
                    action = action_list[i]
                    new_expression = new_expression_list[i] #dynet call for getting element
                    meta_info = meta_info_list[i]
                    new_state = state.get_new_state_after_action(action,meta_info)

                    new_state.path_score_expression = cur_path_expression + new_expression
                    new_state.score = new_state.path_score_expression.scalar_value() - estimated_reward_list[i] #TODO this forward can be expensive...
                    #print("comparison",new_state.score,new_state.path_score_expression.value())
                    #assert abs(new_state.score - new_state.path_score_expression.value()) < 1e-5

                    self.next_states.append(new_state)

            self.next_states.sort(key=lambda x: -1 * x.score)
            # for x in self.next_states:
            #     print("===>",x,x.score)

            next_size = min([len(self.next_states),self.beam_size])

            if contain_nonend_state == False:
                return self.next_states[:next_size]
            else:
                self.cur_save_states = self.next_states[:next_size]

    def beam_find_actions_with_answer_guidence(self,init_state,gold_ans):
        """ perform beam search to find the state with the best reward: maximize \sum R(a) using estimated reward

        :param state: state
        :param state: the gold answers (used to calcuate estimated_reward, also used it to cut the action space)
        :returns: a list of the state (usually with size beam)
        :rtype: list(State)

        """

        init_state.path_score_expression = dt.scalarInput(0)
        init_state.score = 0
        self.cur_save_states = [init_state]

        #print ("*" * 100)
        while True:
            self.next_states = []
            contain_nonend_state = False

            for state in self.cur_save_states:
                cur_path_expression = state.path_score_expression
                cur_path_score = cur_path_expression.scalar_value()

                if state.is_end():
                    self.next_states.append(state) # keep end state for once so that they can compete with longer sequence
                    continue

                contain_nonend_state = True

                action_list, estimated_reward_list = state.get_action_set_withans(gold_ans)
                if action_list == []:
                    continue

                new_expression_list, meta_info_list = state.get_next_score_expressions(action_list)

                for i in range(len(action_list)):
                    action = action_list[i]
                    new_expression = new_expression_list[i] #dynet call for getting element
                    meta_info = meta_info_list[i]
                    new_state = state.get_new_state_after_action(action,meta_info)
                    new_state.path_score_expression = cur_path_expression + new_expression
                    new_state.score = estimated_reward_list[i]

                    self.next_states.append(new_state)

            self.next_states.sort(key=lambda x: (-1 * x.score,-1* (x.path_score_expression.value()))) #sort by reward first; if equal; sort by model score

            next_size = min([len(self.next_states),self.beam_size])

            if contain_nonend_state == False:
                return self.next_states[:next_size]
            else:
                self.cur_save_states = self.next_states[:next_size]


    def beam_train_expected_reward(self,init_state, gold_ans):

        end_state_list = self.beam_predict(init_state)

        # find the best state in the list to make things faster
        reward_states = self.beam_find_actions_with_answer_guidence(init_state, gold_ans)
        if reward_states == []:
            return 0
        best_reward_state = reward_states[0]
        find_best_state = False
        for state in end_state_list:
            if state.action_history == best_reward_state.action_history: #??
                find_best_state = True
                #print("found gold state", find_best_state, best_reward_state)
                break
        if not find_best_state:
            end_state_list.append(best_reward_state)

        exp_path_list = [dt.exp(x.path_score_expression) for x in end_state_list]
        sum_exp = dt.esum(exp_path_list)
        # print("size of end state", len(end_state_list))
        reward_list = [dt.scalarInput(st.reward(gold_ans)) * exp_expr for st ,exp_expr in zip(end_state_list,exp_path_list)]
        # print("=" * 80)
        # for x in end_state_list:
        #      print(x, dt.exp(x.path_score_expression).value(), x.reward(gold_ans))


        expected_negative_reward = -dt.cdiv(dt.esum(reward_list),sum_exp)

        #print("values", sum_exp.value(), (dt.esum(reward_list)).value(), dt.cdiv(dt.esum(reward_list),sum_exp).value())
        # print("=" * 80)
        value = expected_negative_reward.scalar_value()
        #print("obj", expected_negative_reward.value())
        expected_negative_reward.backward()
        self.neural_model.learner.update()
        return value
        # there might be a bug here; should not update at the first iter (if all reward are zero), should not update?

    def beam_train_max_margin_with_answer_guidence(self, init_state, gold_ans):
        # perform two beam search; one for prediction and the other for state action suff
        # max reward y = argmax(r(y)) with the help of gold_ans
        # max y' = argmax f(x,y) - R(y')
        # loss = max(f(x,y') - f(x,y) + R(y) - R(y') , 0)

        #end_state_list = self.beam_predict(init_state)
        end_state_list = self.beam_predict_max_violation(init_state, gold_ans) # have to use this to make it work....
        reward_list = [x.reward(gold_ans) for x in end_state_list]
        violation_list = [s.path_score_expression.value() - reward for s,reward in zip(end_state_list,reward_list)]

        best_score_state_idx = violation_list.index(max(violation_list)) # find the best scoring seq with minimal reward
        best_score_state = end_state_list[best_score_state_idx]
        best_score_state_reward = reward_list[best_score_state_idx]

        loss_value = 0

        if self.only_one_best:
            best_states = self.beam_find_actions_with_answer_guidence(init_state, gold_ans)
            if best_states == []:
                return 0,[]
            best_reward_state = best_states[0]
            #print ("debug: found best_reward_state: qid =", best_reward_state.qinfo.seq_qid, best_reward_state)
            best_reward_state_reward = best_reward_state.reward(gold_ans)
            #print ("debug: best_reward_state_reward =", best_reward_state_reward)
            loss = dt.rectify(best_score_state.path_score_expression - best_reward_state.path_score_expression + dt.scalarInput(best_reward_state_reward - best_score_state_reward))
        else:
            best_states = self.beam_find_actions_with_answer_guidence(init_state, gold_ans)
            best_states_rewards = [s.reward(gold_ans) for s in best_states]
            max_reward = max(best_states_rewards)
            best_states = [s for s,r in zip(best_states,best_states_rewards) if r == max_reward]
            loss = dt.average([dt.rectify(best_score_state.path_score_expression - best_reward_state.path_score_expression + dt.scalarInput(max_reward - best_score_state_reward)) for best_reward_state in best_states])

        loss_value = loss.value()
        loss.backward()

        self.neural_model.learner.update()

        #print ("debug: beam_train_max_margin_with_answer_guidence done. loss_value =", loss_value)

        return loss_value,best_states

    def beam_train_max_margin(self, init_state, gold_ans):
        #still did not use the gold sequence but use the min risk training
        #max reward y = argmax(r(y))
        #max y' = argmax f(x,y) - R(y')
        # loss = max(f(x,y') - f(x,y) + R(y) - R(y') , 0)

        end_state_list = self.beam_predict(init_state)
        reward_list = [x.reward(gold_ans) for x in end_state_list]
        violation_list = [s.score - reward for s,reward in zip(end_state_list,reward_list)]

        best_score_state_idx = violation_list.index(max(violation_list)) # find the best scoring seq with minimal reward
        best_reward_state_idx = reward_list.index(max(reward_list)) # find seq with the max reward in beam

        best_score_state = end_state_list[best_score_state_idx]
        best_reward_state = end_state_list[best_reward_state_idx]

        best_score_state_reward = reward_list[best_score_state_idx]
        best_reward_state_reward = reward_list[best_reward_state_idx]


        loss = dt.rectify(best_score_state.path_score_expression - best_reward_state.path_score_expression + dt.scalarInput(best_reward_state_reward - best_score_state_reward))
        loss_value = loss.value()

        loss.backward()

        self.neural_model.learner.update()
        return loss_value

        #print("loss_value", loss_value)
        #print(self.neural_model.learner.status())
        #for i,ss in enumerate(end_state_list):
        #    print(i, ss, "score", ss.path_score_expression.value(), "reward", reward_list[i])

        #print("first", best_score_state, "score", best_score_state.path_score_expression.value(), "reward", reward_list[best_score_state_idx])
        #print("gold", best_reward_state, "score", best_reward_state.path_score_expression.value(), "reward", reward_list[best_reward_state_idx])


    def beam_train_max_margin_with_goldactions(self,init_state, gold_actions):
        #max y = gold y
        #max y' = argmax f(x,y)
        # loss = max(f(x,y') - f(x,y) + R(y) - R(y') , 0)

        #loss
        #end_state_list = self.beam_predict(init_state)  # top-k argmax_y f(x,y)
        end_state_list = self.beam_predict_max_violation(init_state,gold_actions) # top-k argmax_y f(x,y) + R(y*) - R(y)  // Current implementation is the same as Hamming distance
        best_score_state = end_state_list[0]
        reward_list = [x.reward(gold_actions) for x in end_state_list]

        best_reward_state = self.get_goldstate_with_gold_actions(init_state,gold_actions)
        best_reward = best_reward_state.reward(gold_actions)

        loss = dt.rectify(best_score_state.path_score_expression - best_reward_state.path_score_expression + dt.scalarInput(best_reward-reward_list[0])  )
        loss_value = loss.value()

        loss.backward()
        self.neural_model.learner.update()
        return loss_value


    def greedy_train_max_sumlogllh(self,init_state, gold_actions):

        total_obj = dt.scalarInput(0)

        cur_state = init_state
        res = 0
        idx =  0
        while True:
            if cur_state.is_end():
                break

            action_list = list(cur_state.get_action_set())
            new_expression_list, meta_info_list = cur_state.get_next_score_expressions(action_list)
            prob_list = dt.softmax(new_expression_list)
            gold_action = gold_actions[idx]
            action_idx = action_list.index(gold_action)
            total_obj += -(dt.log(prob_list[action_idx]))

            cur_state = cur_state.get_new_state_after_action(gold_action,meta_info_list[action_idx])
            idx += 1
            #print (cur_state)

        res = total_obj.scalar_value()
        total_obj.backward()
        self.neural_model.learner.update()


        return res

    def get_goldstate_with_gold_actions(self,state,gold_actions):

        cur_state = state.clone()
        cur_state.path_score_expression = dt.scalarInput(0)

        time_idx = 0
        while True:
            if cur_state.is_end():
                break
            action_list = list(cur_state.get_action_set())
            old_expression = cur_state.path_score_expression
            new_expression_list, meta_info_list = state.get_next_score_expressions(action_list)
            gold_act = gold_actions[time_idx]
            action_idx = action_list.index(gold_act)

            cur_state = cur_state.get_new_state_after_action(gold_act,meta_info_list[action_idx])
            cur_state.path_score_expression = old_expression + new_expression_list[action_idx]

            time_idx += 1

        return cur_state


    def greedy_predict(self,state):

        cur_state = state

        while True:
            if cur_state.is_end():
                break

            action_list = cur_state.get_action_set()
            new_expression_list,meta_info_list = cur_state.get_next_score_expressions(action_list)
            prob_list = dt.softmax(new_expression_list)
            pred = np.argmax(prob_list.npvalue())
            action = action_list[pred]

            cur_state = cur_state.get_new_state_after_action(action,meta_info_list[pred])

        return cur_state


if __name__ == '__main__':
    print("test")
