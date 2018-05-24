from sys import float_info

import util, config
from util import QuestionInfo, ResultInfo

class Condition:
    OpGT, OpLT, OpGE, OpLE, OpEqRow, OpNeRow, OpArgMin, OpArgMax = xrange(8)

    def __init__(self, col=-1, op=-1, arg=None):
        self.cond_col = col
        self.operator = op
        self.arg = arg
        return

    @staticmethod
    def numValue(entry, annEntry, default):
        ret = default
        if annEntry.date != None:
            ret = float(annEntry.date.toordinal())
        elif annEntry.number != config.NaN:
            ret = annEntry.number
        elif annEntry.num2 != config.NaN:
            ret = annEntry.num2
        elif util.is_float_try(entry):
            ret = float(entry)
        return ret

    # Check which entries satisfy the condition
    def check(self, qinfo, subtab_rows):
        entries = qinfo.entries
        annTab = qinfo.annTab
        if self.operator == Condition.OpEqRow:
            cond_row = self.arg
            #print("debug: cond_row, self.cond_col = ", cond_row, self.cond_col)
            cond_val = entries[cond_row][self.cond_col].lower()

            try:
                ret = set([r for r in subtab_rows if entries[r][self.cond_col].lower() == cond_val])
            except:
                print("debug: qinfo", qinfo.seq_qid, qinfo.table_file)
                print("debug: cond_row, self.cond_col = ", cond_row, self.cond_col)
                print("debug: subtab_rows", subtab_rows)
                print("debug: entries", entries)

            return ret
        elif self.operator == Condition.OpNeRow:
            cond_row = self.arg
            cond_val = entries[cond_row][self.cond_col].lower()
            return set([r for r in subtab_rows if entries[r][self.cond_col].lower() != cond_val])
        elif self.operator == Condition.OpGT:
            cond_val = self.arg
            return set([r for r in subtab_rows if self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.min) > cond_val])
        elif self.operator == Condition.OpGE:
            cond_val = self.arg
            return set([r for r in subtab_rows if self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.min) >= cond_val])
        elif self.operator == Condition.OpLT:
            cond_val = self.arg
            return set([r for r in subtab_rows if self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.max) < cond_val])
        elif self.operator == Condition.OpLE:
            cond_val = self.arg
            return set([r for r in subtab_rows if self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.max) <= cond_val])
        elif self.operator == Condition.OpArgMin:
            numeric_values = [self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.max) for r in subtab_rows]
            if len(numeric_values) != len(subtab_rows):
                return set()
            min_idx = min((v,i) for i,v in enumerate(numeric_values))[1]
            return [min_idx]
        elif self.operator == Condition.OpArgMax:
            numeric_values = [self.numValue(entries[r][self.cond_col], annTab[(r,self.cond_col)], float_info.min) for r in subtab_rows]
            if len(numeric_values) != len(subtab_rows):
                return set()
            max_idx = max((v,i) for i,v in enumerate(numeric_values))[1]
            return [max_idx]

        assert False, "Unknown condition operator: %d" % self.operator

class Parse:
    Independent, FollowUp = xrange(2)

    def __init__(self):
        self.type = Parse.Independent
        self.select_columns = []    # the list of columns to be selected
        self.conditions = []    # list of Condition objects; meaning that it needs to satisfy all the conditions

    def run(self, qinfo, resinfo):
        if self.type == Parse.Independent:
            if self.select_columns == []:   # if no columns are selected, return all columns
                select_columns = xrange(qinfo.num_columns)
            else:
                select_columns = self.select_columns
            if self.conditions == []:
                legit_rows = set(xrange(qinfo.num_rows))
            else:
                legit_rows = reduce(lambda x,y: x.intersection(y), [cond.check(qinfo, xrange(qinfo.num_rows)) for cond in self.conditions])
        elif self.type == Parse.FollowUp:
            # TODO, ASSUMING THE PREVIOUS ANSWERS ARE SINGLE COLUMN HERE
            ans_col = resinfo.prev_pred_answer_column    # answer column 
            if self.conditions == []:
                # TODO: it should be using the previous answers, but because we're restricted to only one column answers, we have to restrict to that column for now
                # return resinfo.prev_pred_answer_coordinates
                return [coor for coor in resinfo.prev_pred_answer_coordinates if coor[1] == ans_col]
            else:
                legit_rows = reduce(lambda x,y: x.intersection(y), [cond.check(qinfo, resinfo.subtab_rows) for cond in self.conditions])
                select_columns = [ans_col]
        else:
            assert False, "Unknown parse type: %d" % self.type
        return [(r,c) for r in legit_rows for c in select_columns]
