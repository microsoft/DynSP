import sys
from Parse import *
from collections import namedtuple
Action = namedtuple("Action", "type idx row col val")
'''
class Action:
    def __init__(self, act_type, act_idx, row=-1, col=-1, val=-1):
        self.type = act_type    # action type
        self.idx = act_idx      # action index
        self.row = row          # row number -- used by WhereEqRow
        self.col = col          # column number -- used by Select, WhereCol
'''

### Action type:
#
# Original action type:
# (0) START
# (1) STOP (no more condition, 1)
# (2) SELECT X  (# table columns)
# (3) WHERE Y=? (# columns, can be other operators)
# (4) WHERE Y=Z (# rows)
# (5) WHERE Y != Z (# rows)
# (6) WHERE Y > z1 (# values in Q)
# (7) WHERE Y >= z1 (# values in Q)
# (8) WHERE Y < z1 (# values in Q)
# (9) WHERE Y <= z1 (# values in Q)
# (10) WHERE Y is ArgMin (1)
# (11) WHERE Y is ArgMax (1)
#
# Follow-up action type:
# (12) SameAsPrevious (1)
# (13) WHERE Y=? (# columns, can be other operators)
# (14) WHERE Y=Z (# rows in the subtable)
# (15) WHERE Y != Z (# rows in the subtable)
# (16) WHERE Y > z1 (# values in Q)
# (17) WHERE Y >= z1 (# values in Q)
# (18) WHERE Y < z1 (# values in Q)
# (19) WHERE Y <= z1 (# values in Q)
# (20) WHERE Y is ArgMin (1)
# (21) WHERE Y is ArgMax (1)
#
# Legit sequence: Check OneNote for the transition diagram
class ActionType:
    Num_Types = 22
    Start, \
        Stop, Select, WhereCol, CondEqRow, CondNeRow, CondGT, CondGE, CondLT, CondLE, ArgMin, ArgMax, \
        SameAsPrevious, FpWhereCol, FpCondEqRow, FpCondNeRow, FpCondGT, \
        FpCondGE, FpCondLT, FpCondLE, FpArgMin, FpArgMax = xrange(Num_Types)
    LegitNextActionType = {}
    IndepLegitNextActionType = {}

    IndepLegitNextActionType[Start] = [Select]

    LegitNextActionType[Start] = [Select, SameAsPrevious, FpWhereCol]
    LegitNextActionType[Select] = [Stop, WhereCol]
    #LegitNextActionType[Select] = [Stop]
    LegitNextActionType[Stop] = []
    LegitNextActionType[WhereCol] = [CondEqRow, CondNeRow, CondGT, CondGE, CondLT, CondLE, ArgMin, ArgMax]
    LegitNextActionType[CondEqRow] = []
    LegitNextActionType[CondNeRow] = []
    LegitNextActionType[CondGT] = []
    LegitNextActionType[CondGE] = []
    LegitNextActionType[CondLT] = []
    LegitNextActionType[CondLE] = []
    LegitNextActionType[ArgMin] = []
    LegitNextActionType[ArgMax] = []

    LegitNextActionType[SameAsPrevious] = []
    LegitNextActionType[FpWhereCol] = [FpCondEqRow, FpCondNeRow, FpCondGT, FpCondGE, FpCondLT, FpCondLE, FpArgMin, FpArgMax]
    LegitNextActionType[FpCondEqRow] = []
    LegitNextActionType[FpCondNeRow] = []
    LegitNextActionType[FpCondGT] = []
    LegitNextActionType[FpCondGE] = []
    LegitNextActionType[FpCondLT] = []
    LegitNextActionType[FpCondLE] = []
    LegitNextActionType[FpArgMin] = []
    LegitNextActionType[FpArgMax] = []

    # group the action types based on their numbers of instances and more
    SingleInstanceActions = set([Start, Stop, SameAsPrevious, ArgMin, ArgMax, FpArgMin, FpArgMax])
    ColumnActions = set([Select, WhereCol, FpWhereCol])
    RowActions = set([CondEqRow, CondNeRow])
    SubTabRowActions = set([FpCondEqRow, FpCondNeRow])
    QuesValueActions = set([CondGE, CondGT, CondLE, CondLT, FpCondGT, FpCondGE, FpCondLT, FpCondLE])
    
    WhereConditions = set([CondEqRow, CondNeRow, CondGE, CondGT, CondLE, CondLT, ArgMin, ArgMax])
    FpWhereConditions = set([FpCondEqRow, FpCondNeRow, FpCondGT, FpCondGE, FpCondLT, FpCondLE, FpArgMin, FpArgMax])
    Conditions = WhereConditions.union(FpWhereConditions)
    FirstQuestionActions = set([Start, Stop, Select, WhereCol, CondEqRow, CondNeRow, CondGT, CondGE, CondLT, CondLE, ArgMin, ArgMax])
    FollowUpQuestionActions = set([Start, Stop, SameAsPrevious, FpWhereCol, FpCondEqRow, FpCondNeRow, FpCondGT, \
                                    FpCondGE, FpCondLT, FpCondLE, FpArgMin, FpArgMax])
    if config.d["ReduceRowCond"]:
        EqRowConditions = set([CondEqRow, CondNeRow, FpCondEqRow, FpCondNeRow])
    else:
        EqRowConditions = set([])

    NumericConditions = QuesValueActions.union(set([ArgMin, ArgMax, FpArgMin, FpArgMax]))

class ActionFactory:
    def __init__(self, qinfo, resinfo=None):
        self.qinfo = qinfo
        self.actions = []
        self.type2actidxs = {}
        self.legit_next_action_idxs_cache = {}
        self.legit_next_action_idxs_history_cache = {}
        if resinfo == None:
            self.subtab_rows = []
        else:
            self.subtab_rows = resinfo.subtab_rows

        #print("qinfo.question:", qinfo.question)
        #print("qinfo.values_in_ques:", qinfo.values_in_ques)

        # Create "instances" of action types; essentially list the number of the actions each type can have
        p = 0
        for act_type in xrange(ActionType.Num_Types):
            # number of instances: 1
            if act_type in ActionType.SingleInstanceActions:
                num_acts = self.append_actions(act_type, p)
            # number of instances: # columns
            elif act_type in ActionType.ColumnActions:
                num_acts = self.append_actions(act_type, p, qinfo.num_columns, cols = xrange(qinfo.num_columns))
            # number of instances: # rows
            elif act_type in ActionType.RowActions:
                num_acts = self.append_actions(act_type, p, qinfo.num_rows, rows = xrange(qinfo.num_rows))
            # number of instances: # rows in subtable
            elif act_type in ActionType.SubTabRowActions:
                num_acts = self.append_actions(act_type, p, len(self.subtab_rows), rows = self.subtab_rows)
            # number of instances: # values in question
            elif act_type in ActionType.QuesValueActions:
                #print("qinfo.values_in_ques = ", qinfo.values_in_ques, len(qinfo.values_in_ques))
                num_acts = self.append_actions(act_type, p, len(qinfo.values_in_ques), values=qinfo.values_in_ques)

            self.type2actidxs[act_type] = xrange(p, p+num_acts)
            p += num_acts
        
        # special action -- Start
        self.start_action_idx = self.type2actidxs[ActionType.Start][0]

    def append_actions(self, act_type, start_idx, num_act = 1, rows = None, cols = None, values = None):
        if num_act == 1 and rows == None and cols == None and values == None: # do not need either row or column info
            self.actions.append(Action(act_type, start_idx, -1, -1, -1))
        elif rows == None and cols != None and values == None: # need column info
            assert(num_act == len(cols))
            for p in xrange(num_act):
                act_idx = start_idx + p
                self.actions.append(Action(act_type, act_idx, -1, cols[p], -1))
        elif rows != None and cols == None and values == None: # need row info
            assert(num_act == len(rows))
            for p in xrange(num_act):
                act_idx = start_idx + p
                self.actions.append(Action(act_type, act_idx, rows[p], -1, -1))
        elif rows == None and cols == None and values != None: # need values in the question
            assert(num_act == len(values))
            for p,v in enumerate(values):
                act_idx = start_idx + p
                self.actions.append(Action(act_type, act_idx, -1, -1, v))
        else:
            print("Debug: rows = ", rows, "cols = ", cols, "values = ", values)
            assert False, "Error! Unknown act_type: %d" % act_type
        return num_act

    def legit_next_action_idxs(self, act_idx, action_history = None):

        # check history cache
        if action_history != None:
            action_history_key = ','.join(map(str,action_history))
            if action_history_key in self.legit_next_action_idxs_history_cache:
                return self.legit_next_action_idxs_history_cache[action_history_key]

        # The first chunk of this function determines the possible next actions based only on the given current action.
        # In other words, it only checks whether the transition is defined previously in the action-space graph.
        if act_idx not in self.legit_next_action_idxs_cache:
            #print ("debug:", act_idx, self.actions)
            act = self.actions[act_idx]
            act_type = act.type
            if not self.subtab_rows and act_type in ActionType.IndepLegitNextActionType:  # no previous question, and special state 
                legit_next_action_types = ActionType.IndepLegitNextActionType[act_type]
            else:
                legit_next_action_types = ActionType.LegitNextActionType[act_type]
            ret = []
            for legit_type in legit_next_action_types:
                if legit_type in ActionType.EqRowConditions:    # remove some equivalent equal row conditions (changing the action to condition on value directly is difficult given the current code design...)
                    addedCondValues = set()
                    for legit_act_idx in self.type2actidxs[legit_type]:
                        act = self.actions[legit_act_idx]
                        entVal = self.qinfo.entries[act.row][act.col]
                        if entVal in addedCondValues:
                            continue
                        addedCondValues.add(entVal)
                        ret.append(legit_act_idx)
                elif legit_type in ActionType.NumericConditions:
                    cond_col = self.actions[act_idx].col
                    if cond_col not in self.qinfo.numeric_cols:  # not a numeric column
                        continue

                    #print ("debug: passed numeric columns", self.qinfo.table_file, cond_col)
                    for legit_act_idx in self.type2actidxs[legit_type]:
                        ret.append(legit_act_idx)
                else:
                    for legit_act_idx in self.type2actidxs[legit_type]:
                        ret.append(legit_act_idx)
            self.legit_next_action_idxs_cache[act_idx] = ret
        else:
            ret = self.legit_next_action_idxs_cache[act_idx]

        # The second chunk of this function "prunes" some of the possible actions by looking at the existing partial parse,
        # as well as taking the cues from the question. The goal is to reduce the search space whenever possible.
        if action_history != None:  # Check if actions that have to be unique already occur
            act_set = set(action_history)
            new_ret = []
            for act_idx in ret:
                act = self.actions[act_idx]
                if (act.type in ActionType.Conditions) and (act_idx in act_set):  # no duplicate conditions
                    continue
                if (self.qinfo.pos == 0) and (act.type not in ActionType.FirstQuestionActions):   # first question
                    continue
                # follow-up question with keywords indicating it's a dependent parse
                # TODO: unify the code in action_history_quality
                if self.qinfo.pos != 0 \
                    and (self.qinfo.contain_ngram('of those') or self.qinfo.contain_ngram('which one') or self.qinfo.contain_ngram('which ones')) \
                    and (act.type not in ActionType.FollowUpQuestionActions):
                    continue
                new_ret.append(act_idx)

            #if new_ret == []:
            #    print (self.qinfo.seq_qid, self.qinfo.question, action_history, ret)
            ret = new_ret
            self.legit_next_action_idxs_history_cache[action_history_key] = ret

        #print("legit_next_action_idxs", "input:", act_idx, "output:", ret)

        return ret

    def find_actions(self, act_idxs, act_type):
        return [self.actions[act_idx] for act_idx in act_idxs if self.actions[act_idx].type == act_type]

    # map a sequence of actions to a parse
    def action_history_to_parse(self, act_idxs):

        #print("act_idxs", act_idxs)

        parse = Parse()

        p, act_history_length = 0, len(act_idxs)
        while p < act_history_length:
            act_idx = act_idxs[p]
            act = self.actions[act_idx]
            if act.type == ActionType.Select:   # having SELECT meaning it's an independent parse
                parse.type = Parse.Independent
                parse.select_columns.append(act.col)    # record the column it selects
            elif act.type == ActionType.WhereCol:
                col = act.col
                # have to assume that the follow-up action is about the operator and argument
                p += 1
                #print("p =", p)
                act_idx = act_idxs[p]

                act = self.actions[act_idx]

                if act.type != ActionType.Stop:
                    # This is the only legitimate type after WhereCol currently.  Will have to expand the coverage later for more action types.
                    assert (act.type in ActionType.WhereConditions), "Illegit action type after WhereCol: %d" % act.type
                    if act.type == ActionType.CondEqRow:
                        cond = Condition(col, Condition.OpEqRow, act.row)
                    elif act.type == ActionType.CondNeRow:
                        cond = Condition(col, Condition.OpNeRow, act.row)
                    elif act.type == ActionType.CondGT:
                        cond = Condition(col, Condition.OpGT, act.val[1])
                    elif act.type == ActionType.CondGE:
                        cond = Condition(col, Condition.OpGE, act.val[1])
                    elif act.type == ActionType.CondLT:
                        cond = Condition(col, Condition.OpLT, act.val[1])
                    elif act.type == ActionType.CondLE:
                        cond = Condition(col, Condition.OpLE, act.val[1])
                    elif act.type == ActionType.ArgMin:
                        cond = Condition(col, Condition.OpArgMin)
                    elif act.type == ActionType.ArgMax:
                        cond = Condition(col, Condition.OpArgMax)
                    parse.conditions.append(cond)
            elif act.type == ActionType.SameAsPrevious:
                parse.type = Parse.FollowUp
            elif act.type == ActionType.FpWhereCol:
                parse.type = Parse.FollowUp
                col = act.col
                # have to assume that the follow-up action is about the operator and argument
                p += 1
                act_idx = act_idxs[p]
                act = self.actions[act_idx]

                if act.type != ActionType.Stop:
                    # This is the only legitimate type after WhereCol currently.  Will have to expand the coverage later for more action types.
                    assert (act.type in ActionType.FpWhereConditions), "Illegit action type after FpWhereCol: %d" % act.type
                    if act.type == ActionType.FpCondEqRow:
                        cond = Condition(col, Condition.OpEqRow, act.row)
                    elif act.type == ActionType.FpCondNeRow:
                        cond = Condition(col, Condition.OpNeRow, act.row)
                    elif act.type == ActionType.FpCondGT:
                        cond = Condition(col, Condition.OpGT, act.val[1])
                    elif act.type == ActionType.FpCondGE:
                        cond = Condition(col, Condition.OpGE, act.val[1])
                    elif act.type == ActionType.FpCondLT:
                        cond = Condition(col, Condition.OpLT, act.val[1])
                    elif act.type == ActionType.FpCondLE:
                        cond = Condition(col, Condition.OpLE, act.val[1])
                    elif act.type == ActionType.FpArgMin:
                        cond = Condition(col, Condition.OpArgMin)
                    elif act.type == ActionType.FpArgMax:
                        cond = Condition(col, Condition.OpArgMax)
                    parse.conditions.append(cond)
            else:
                assert (act.type == ActionType.Start or act.type == ActionType.Stop), "Unknown action type: %d" % act.type
            p += 1

        return parse

    def action_history_quality(self, act_idxs):
        # if it's a follow-up question & contains "of those" or "which", then it's forced to switch "follow up" actions
        if self.qinfo.pos != 0 \
            and (self.qinfo.contain_ngram('of those') or self.qinfo.contain_ngram('which one') or self.qinfo.contain_ngram('which ones')):
            parse = self.action_history_to_parse(act_idxs)  # TODO: action_history_to_parse has been called multiple times
            #print ("\t".join([self.actidx2str(act) for act in act_idxs]))
            if parse.type == Parse.Independent:
                return 0.0

        return 1.0

    # for debugging
    def actidx2str(self, act_idx):
        action = self.actions[act_idx]
        if action.type == ActionType.Start:
            #return "START"
            return ""
        elif action.type == ActionType.Stop:
            #return "Stop"
            return ""
        elif action.type == ActionType.Select:
            col = action.col
            return "SELECT %s" % self.qinfo.headers[col]
        elif action.type == ActionType.WhereCol:
            col = action.col
            return "WHERE %s" % self.qinfo.headers[col]
        elif action.type == ActionType.CondEqRow:
            return "= ROW %d" % action.row
        elif action.type == ActionType.CondNeRow:
            return "!= ROW %d" % action.row
        elif action.type == ActionType.CondGT:
            return "> %f" % action.val[1]
        elif action.type == ActionType.CondGE:
            return ">= %f" % action.val[1]
        elif action.type == ActionType.CondLT:
            return "< %f" % action.val[1]
        elif action.type == ActionType.CondLE:
            return "<= %f" % action.val[1]
        elif action.type == ActionType.ArgMin:
            return "is Min"
        elif action.type == ActionType.ArgMax:
            return "is Max"
        elif action.type == ActionType.SameAsPrevious:
            return "SameAsPrevious"
        elif action.type == ActionType.FpWhereCol:
            col = action.col
            return "FollowUp WHERE %s" % self.qinfo.headers[col]
        elif action.type == ActionType.FpCondEqRow:
            return "= ROW %d" % action.row
        elif action.type == ActionType.FpCondNeRow:
            return "!= ROW %d" % action.row
        elif action.type == ActionType.FpCondGT:
            return "> %f" % action.val[1]
        elif action.type == ActionType.FpCondGE:
            return ">= %f" % action.val[1]
        elif action.type == ActionType.FpCondLT:
            return "< %f" % action.val[1]
        elif action.type == ActionType.FpCondLE:
            return "<= %f" % action.val[1]
        elif action.type == ActionType.FpArgMin:
            return "is Min"
        elif action.type == ActionType.FpArgMax:
            return "is Max"
        else:
            assert False, "Error! Unknown action.type: %d" % action.type