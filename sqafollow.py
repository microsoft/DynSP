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
from sqastate import *
from sqamodel import *
import nnmodule as nnmod

######## START OF THE CODE ########

'''
This is forked from "sqafirst.py" and is designed for "follow-up" questions.
In addition to the regular input, question and table, it also has:
    (1) the previous question
    (2) the parse of the previous question
    (3) the results of the previous question
In other words, the action space is bigger in the sense that it can ignore the previous question completely,
or it can take the previous question into account.
'''


def test_model(test, search_manager, model, fLog=None, fnRes='', blWTQ=False, indep=False):
    if fnRes:
        if blWTQ:
            fRes = open(fnRes, 'w')
        else:
            fieldnames = ['id', 'annotator', 'position', 'answer_coordinates']
            fRes = csv.DictWriter(open(fnRes, 'w'), delimiter=str('\t'), fieldnames=fieldnames)
            fRes.writeheader()

    accuracy = 0.0
    all_reward = 0.0
    total = 0.0
    for i, qinfo in enumerate(test, 1):
        dt.renew_cg()  # very important! to renew the cg

        if qinfo.seq_qid[-1] == '0' or indep:  # first question or treat all questions independently
            init_state = SqaScoreExpressionState(model, qinfo, testmode=True)
        else:
            init_state = SqaScoreExpressionState(model, qinfo, resinfo=resinfo, testmode=True)

        top_states = search_manager.beam_predict(init_state)
        top1_state = top_states[0]
        rew = top1_state.reward(qinfo.answer_coordinates)
        pred = top1_state.recent_pred

        # use the predicted answers & the most common column index

        if pred == []:
            # empty answers, set pred_column_idx to be 0, as it does not matter...
            resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence, pred, 0)
        else:
            pred_columns = [coord[1] for coord in pred]
            pred_column_idx = Counter(pred_columns).most_common(1)[0][0]
            resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence,
                                      pred, pred_column_idx)
        # output to result file
        if fnRes:
            if blWTQ:
                id, annotator, position = qinfo.seq_qid.split('_')
                fRes.write(id)
                for coord in pred:
                    fRes.write("\t%s" % qinfo.entries[coord[0]][coord[1]])
                fRes.write('\n')
            else:
                outRow = {}
                outRow['id'], outRow['annotator'], outRow['position'] = qinfo.seq_qid.split('_')
                outRow['answer_coordinates'] = "[%s]" % ", ".join(["'(%d, %d)'" % coord for coord in pred])
                fRes.writerow(outRow)

        # output detailed predictions
        if fLog:
            fLog.write("(%s) %s\n" % (qinfo.seq_qid, qinfo.question))
            fLog.write("%s/%s\n" % (config.d["dirTable"], qinfo.table_file))
            if config.d["guessLogPass"]:
                best_reward_state = \
                search_manager.beam_find_actions_with_answer_guidence(init_state, qinfo.answer_coordinates)[0]
                fLog.write("Guessed Gold Parse: %s\n" % best_reward_state)

            if config.d["verbose-dump"]:
                for i, s in enumerate(top_states, 1):
                    fLog.write("Parse %d: [%f] %s\n" % (i, s.score, s))
            else:
                fLog.write("Parse: %s\n" % top1_state)

            fLog.write("Answer: %s\n" % ", ".join(["(%d,%d)" % coord for coord in qinfo.answer_coordinates]))
            fLog.write("Predictions: %s\n" % ", ".join(["(%d,%d)" % coord for coord in pred]))
            fLog.write("Reward: %f\n" % rew)
            fLog.write("Accuracy: %d\n" % int(rew))
            fLog.write("\n")

        all_reward += rew
        accuracy += int(rew)  # 0-1, only get a point if all predictions are correct
        total += 1

    reported_reward = all_reward / total
    reported_accuracy = accuracy / total

    if fLog:
        fLog.write("Average reward: %f\n" % reported_reward)
        fLog.write("Average accuracy %f\n" % reported_accuracy)

    if fnRes and blWTQ:
        fRes.close()

    return reported_reward, reported_accuracy


def main():
    parser = argparse.ArgumentParser(description='Targeting "first questions" only.')
    parser.add_argument('--expSym',
                        help='1, 2, 3, 4, 5 or 0 (full), or 11 (WTQ), 21 (WTQ-dev1) .. 25 (WTQ-dev5), or -1 (quick test)',
                        type=int)
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-seed')
    parser.add_argument('--log', help='file to store additional log information', default='')
    parser.add_argument('--res', help='file to store the predictions on test set in the "official" format', default='')
    parser.add_argument('--model', help='prefix of the file that stores the model parameters', default='')
    parser.add_argument('--firstOnly', help='load only first questions', action='store_const', const=True,
                        default=False)
    parser.add_argument('--dirData', help='data folder', default='data')
    parser.add_argument('--evalModel', help='evaluate a particular model on the test data only', default='')
    parser.add_argument('--evalOracle', help='yes/no switch with see oracle results for the action space',
                        action='store_const', const=True, default=False)
    parser.add_argument('--wtq', help='WikiTableQuestions output', action='store_const', const=True, default=False)
    parser.add_argument('--indep', help='Treat all questions independently', action='store_const', const=True,
                        default=False)

    args = parser.parse_args()

    # Prepare training and testing (development) data
    random.seed(1)

    data_folder = args.dirData
    # config.d["AnnotatedTableDir"] = args.dirData.replace('data', config.d["AnnotatedTableDir"])
    if args.expSym == 0:
        print("Full Train/Test splits...")
        train_file = "%s/train.tsv" % data_folder
        test_file = "%s/test.tsv" % data_folder
    elif args.expSym in xrange(1, 6):
        print("Random-split-%d-train/dev..." % args.expSym)
        train_file = "%s/random-split-%d-train.tsv" % (data_folder, args.expSym)
        test_file = "%s/random-split-%d-dev.tsv" % (data_folder, args.expSym)
    elif args.expSym == -1:
        print("Quick code test...")
        train_file = "%s/unit.tsv" % data_folder
        test_file = "%s/unit.tsv" % data_folder
    elif args.expSym == 11:
        print("WikiTable Questions...")
        train_file = "%s/training.tsv" % data_folder
        test_file = "%s/pristine-unseen-tables.tsv" % data_folder
    elif args.expSym in xrange(21, 26):
        print("WikiTable Questions -- Dev1/Dev1...")
        train_file = "%s/random-split-%d-dev.tsv" % (data_folder, args.expSym - 20)
        test_file = "%s/random-split-%d-dev.tsv" % (data_folder, args.expSym - 20)
    else:
        assert False, "Unknown experimental setting..."
        return

    if args.evalModel:
        evalModel(args.evalModel, data_folder, test_file, args.log, args.res, args.wtq, args.indep)
        return

    fLog = None
    if args.log:
        fLog = open(args.log, 'w')

    print("=" * 80)
    print("Train", train_file)
    print("Test", test_file)
    print(config.d)
    print(">" * 8, "begin experiments")

    train = util.get_labeled_questions(train_file, data_folder, args.firstOnly)
    test = util.get_labeled_questions(test_file, data_folder, args.firstOnly)

    # create a word embedding table

    words = set(["_UNK_", "_EMPTY_"])
    for ex in train:
        words.update(ex.all_words)
    for ex in test:
        words.update(ex.all_words)

    vw = util.Vocab.from_corpus([words])
    nwords = vw.size()

    neural_model = SqaModel(0.01, vw)
    sm = BeamSearchInferencer(neural_model, config.d["beam_size"])
    sm.only_one_best = config.OnlyOneBest

    if args.evalOracle:
        evalOracleActions(neural_model, sm, train)
        return

    # main loop
    start_time = time.time()
    max_reward_at_epoch = (0, 0, 0)
    max_accuracy_at_epoch = (0, 0, 0)
    for ITER in xrange(config.d["NUM_ITER"]):
        # random.shuffle(train)
        loss = 0
        for i, qinfo in enumerate(train, 1):
            dt.renew_cg()  # very important! to renew the cg

            if qinfo.seq_qid[-1] == '0' or args.indep:  # first question or treat all questions independenly
                init_state = SqaScoreExpressionState(neural_model, qinfo)
            else:
                init_state = SqaScoreExpressionState(neural_model, qinfo, resinfo=resinfo)

            # loss += sm.beam_train_max_margin(init_state, qinfo.answer_coordinates)
            try:
                new_loss, end_state_list = sm.beam_train_max_margin_with_answer_guidence(init_state,
                                                                                         qinfo.answer_coordinates)
                loss += new_loss

                if random.uniform(0, 1) < 0.5:
                    # print("use gold!")
                    # use the gold answers
                    resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence,
                                              qinfo.answer_coordinates, qinfo.answer_column_idx)
                else:
                    # print("use predict!")

                    if len(end_state_list) == 0:
                        # empty answers, set pred_column_idx to be 0, as it does not matter...
                        resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence, pred, 0)
                    else:
                        # use the predicted answers & the most common column index

                        pred = end_state_list[0].recent_pred
                        # use the predicted answers & the most common column index
                        if pred == []:
                            resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence, pred, 0)
                        else:
                            pred_columns = [coord[1] for coord in pred]
                            pred_column_idx = Counter(pred_columns).most_common(1)[0][0]
                            resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence,
                                                      pred, pred_column_idx)
            except Exception as e:
                print(str(e))
                print("Exception in running!")

            # print ("debug: resinfo:", resinfo)

            if i % 100 == 0:
                print(i, "/", len(train))
                # print ("debug: dynet.parameter(sm.neural_model.SelColW).value():", dt.parameter(sm.neural_model.SelColW).value())
                # print ("debug: loss:", loss)

        neural_model.learner.update_epoch(1.0)
        print("In epoch ", ITER, " avg loss (or negative reward) is ", loss / len(train))
        reported_reward, reported_accuracy = test_model(test, sm, neural_model, fLog, indep=args.indep)
        print("In epoch ", ITER, " test reward is %f, test accuracy is %f" % (reported_reward, reported_accuracy))

        if (reported_reward > max_reward_at_epoch[0]):
            max_reward_at_epoch = (reported_reward, reported_accuracy, ITER)

        if (reported_accuracy > max_accuracy_at_epoch[0]):
            max_accuracy_at_epoch = (reported_accuracy, reported_reward, ITER)

        now_time = time.time()
        print("Time taken in this epoch", now_time - start_time)
        start_time = now_time
        print("Best Reward: %f (Accuracy: %f) at epoch %d" % max_reward_at_epoch)
        print("Best Accuracy: %f (Reward: %f) at epoch %d" % max_accuracy_at_epoch)

        if args.model:
            neural_model.save("%s-%d" % (args.model, ITER))
            '''
            if fLog:    # Test the saved model
                new_model = SqaModel.load(args.model)
                new_sm = BeamSearchInferencer(new_model,config.d["beam_size"])
                reported_reward,reported_accuracy = test_model(test, new_sm, new_model, fLog)
            '''
        print()
        sys.stdout.flush()

    if args.res: test_model(test, sm, model, fnRes=args.res, blWTQ=args.wtq, indep=args.indep)

    if fLog: fLog.close()


def evalOracleActions(neural_model, sm, train):
    train_reward = 0.0
    count_perfect_reward = 0.0

    first_train_reward = 0.0
    first_count_perfect_reward = 0.0
    num_first = 0.0

    rest_train_reward = 0.0
    rest_count_perfect_reward = 0.0
    num_rest = 0.0

    for i, qinfo in enumerate(train, 1):
        dt.renew_cg()  # very important! to renew the cg

        if qinfo.seq_qid[-1] == '0':  # first question
            init_state = SqaScoreExpressionState(neural_model, qinfo)
        else:
            init_state = SqaScoreExpressionState(neural_model, qinfo, resinfo=resinfo)

        gold_ans = qinfo.answer_coordinates

        top_states = sm.beam_find_actions_with_answer_guidence(init_state, gold_ans)
        best_reward_state = top_states[0]
        best_reward_state_reward = best_reward_state.reward(gold_ans)

        # output detailed predictions
        sys.stdout.write("(%s) %s\n" % (qinfo.seq_qid, qinfo.question))
        sys.stdout.write("%s/%s\n" % (config.d["dirTable"], qinfo.table_file))
        if config.d["verbose-dump"]:
            for i, s in enumerate(top_states, 1):
                sys.stdout.write("Parse %d: estimate reward: [%f] model score: [%f] %s\n" % (
                i, s.score, s.path_score_expression.value(), s))
        else:
            sys.stdout.write("Parse: %s\n" % best_reward_state)

        sys.stdout.write("Answer: %s\n" % ", ".join(["(%d,%d)" % coord for coord in qinfo.answer_coordinates]))
        sys.stdout.write("\n")

        train_reward += best_reward_state_reward
        if (best_reward_state_reward == 1):
            count_perfect_reward += 1

        if qinfo.seq_qid[-1] == '0':  # first question
            num_first += 1
            first_train_reward += best_reward_state_reward
            if (best_reward_state_reward == 1):
                first_count_perfect_reward += 1
        else:
            num_rest += 1
            rest_train_reward += best_reward_state_reward
            if (best_reward_state_reward == 1):
                rest_count_perfect_reward += 1

        # use the gold answers
        resinfo = util.ResultInfo(qinfo.seq_qid, qinfo.question, qinfo.ques_word_sequence,
                                  qinfo.answer_coordinates, qinfo.answer_column_idx)

        if i % 100 == 0:
            print(i, "/", len(train))
            # print ("debug: dynet.parameter(sm.neural_model.SelColW).value():", dt.parameter(sm.neural_model.SelColW).value())
            # print ("debug: loss:", loss)

    print("With beamsize  ", config.d['beam_size'])
    print("Oracle training reward is ", (train_reward / len(train)))
    print("percentage of getting perfect reward is ", (count_perfect_reward / len(train)))

    print("# first question  ", num_first)
    print("Oracle training reward is ", (first_train_reward / num_first))
    print("percentage of getting perfect reward is ", (first_count_perfect_reward / num_first))

    print("# non-first question  ", num_rest)
    print("Oracle training reward is ", (rest_train_reward / num_rest))
    print("percentage of getting perfect reward is ", (rest_count_perfect_reward / num_rest))


def evalModel(fnModel, data_folder, fnData, fnLog='', fnRes='', blWTQ=False, indep=False):
    data = util.get_labeled_questions(fnData, data_folder, skipEmptyAns=not blWTQ)
    model = SqaModel.load(fnModel)
    sm = BeamSearchInferencer(model, config.d["beam_size"])

    fLog = None
    if fnLog: fLog = open(fnLog, 'w')
    test_model(data, sm, model, fLog=fLog, fnRes=fnRes, blWTQ=blWTQ, indep=indep)
    if fnLog: fLog.close()


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
