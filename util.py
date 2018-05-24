from itertools import count

import argparse, os, cPickle, sys, csv, glob, json, itertools, re, datetime
from unidecode import unidecode
from collections import Counter, defaultdict
from tokenizer import simpleTokenize

import config

def read(fname):
    sent = []
    for line in file(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w,p = line
            sent.append((w,p))

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

class CorpusReader:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in file(self.fname):
            line = line.strip().split()
            #line = [' ' if x == '' else x for x in line]
            yield line

class CharsCorpusReader:
    def __init__(self, fname, begin=None):
        self.fname = fname
        self.begin = begin
    def __iter__(self):
        begin = self.begin
        for line in file(self.fname):
            line = list(line)
            if begin:
                line = [begin] + line
            yield line

##------------------------------------------------------------------------------

class QuestionInfo:
    def __init__(self, seq_qid = "", pos = -1, question = "", table_file = "", headers = [], entries = [], types = [], 
                 answer_column_idx = -1, answer_column_name = "", answer_rows = [], is_col_select = None, 
                 eq_cond_column_idx = -1, eq_cond_column_name = "", eq_cond_value = "",
                 complete_match = None, answer_coordinates = [], answer_text = [], annTab = None, 
                 numeric_cols = set()):

        self.seq_qid = seq_qid               # question id
        self.pos = pos               # question position in a sequence
        self.question = question              # question
        self.table_file = table_file        # table file name
        self.headers = headers               # table header fields
        self.entries = entries               # table content
        self.types = types                 # table column field types
        self.answer_column_idx = answer_column_idx     # answer column index
        self.answer_column_name = answer_column_name    # answer column field
        self.answer_rows = answer_rows           # answer row indices
        self.is_col_select = is_col_select       # is a column-select-only question?
        self.eq_cond_column_idx = eq_cond_column_idx    # column index of Y in (Y=Z condition)
        self.eq_cond_column_name = eq_cond_column_name   # column index of Y in (Y=Z condition)
        self.eq_cond_value = eq_cond_value         # column value of Z in (Y=Z condition)
        self.complete_match = complete_match      # does our "parse" answer the question?
        self.answer_coordinates = answer_coordinates    # answer_coordinates
        self.answer_text = answer_text           # answer_text
        self.annTab = annTab        # annotated table
        self.numeric_cols = numeric_cols    # set of numeric columns

        self.all_words = self.comp_all_words()
        self.ques_word_sequence = self.comp_ques_word_sequence()
        self.ques_word_sequence_ngram_str = ' ' + ' '.join(self.ques_word_sequence) + ' '
        self.headers_word_sequences = self.comp_headers_word_sequences()
        self.entries_word_sequences = self.comp_entries_word_sequences()

        #print("before call...")
        numbers = findNumbers(self.ques_word_sequence)
        #print("after call", numbers, len(numbers))
        #self.values_in_ques = findNumbers(self.ques_word_sequence), # numeric values in the question
        self.values_in_ques = numbers
        
        #print("values_in_ques:", self.values_in_ques, len(self.values_in_ques))

        self.num_rows = len(self.entries)
        self.num_columns = len(self.headers)

    # TODO: call the following functions to make sure "lower" is consistent
    def comp_all_words(self):
        words = set()
        for w in simpleTokenize(self.question): words.add(w.lower())
        for colname in self.headers:
            for w in simpleTokenize(colname): words.add(w.lower())
        for row in self.entries:
            for ent in row:
                for w in simpleTokenize(ent): words.add(w.lower())
        return words

    def comp_ques_word_sequence(self):
        return [w.lower() for w in simpleTokenize(self.question)]

    def comp_headers_word_sequences(self):
        ret = []
        for colname in self.headers:
            ret.append([w.lower() for w in simpleTokenize(colname)])
        return ret

    def comp_entries_word_sequences(self):
        ret = []
        for row in self.entries:
            row_sequences = []
            for ent in row:
                row_sequences.append([w.lower() for w in simpleTokenize(ent)])
            ret.append(row_sequences)
        return ret

    def contain_ngram(self, sub_ngram_str):
        sub_ngram_str = ' ' + sub_ngram_str + ' '
        ret = sub_ngram_str in self.ques_word_sequence_ngram_str
        #print(ret, sub_ngram_str, self.ques_word_sequence_ngram_str)
        return ret
    
class ResultInfo:   # information of the previous question and its parse and answers
    def __init__(self, seq_qid = "", question = "", ques_word_sequence = [], 
                 pred_answer_coordinates = [], pred_answer_column = -1, pred_question_parse = []):
        self.prev_seq_qid = seq_qid                  # previous question id
        self.prev_question = question                 # previous question
        self.prev_ques_word_sequence = ques_word_sequence   # previous question word sequence
        self.prev_pred_answer_coordinates = pred_answer_coordinates  # predicted answer_coordinates of previous question
        self.prev_pred_answer_column = pred_answer_column       # predicted answer column of previous question (i.e., SELECT X)
        self.prev_question_parse = pred_question_parse   # predicted parse of previous question (i.e., the final picked action_history)
        self.subtab_rows = sorted(list(set([coor[0] for coor in pred_answer_coordinates])))
    def __str__(self):
        ret = "prev_seq_qid = '%s', prev_question = '%s', prev_ques_word_sequence = '[%s]'" % (self.prev_seq_qid, self.prev_question, self.prev_ques_word_sequence)
        ret += "prev_pred_answer_column = '%s'" % self.prev_pred_answer_column
        return ret

def get_question_ids(fnTsv, dirData):
	reader = csv.DictReader(open("%s" % (fnTsv), 'r'), delimiter='\t')
	ret = []
	for row in reader:
		ret.append(row['id'])
	return ret

# read questions from dataset and store the question information
def get_labeled_questions(fnTsv, dirData, firstOnly=False, skipEmptyAns=True):
    reader = csv.DictReader(open("%s" % (fnTsv), 'r'), delimiter='\t')
    data = []

    for row in reader:
        qid = row['id']
        count = row['annotator']
        pos = row['position']
        lstLoc = eval(row['answer_coordinates'])
        if skipEmptyAns and lstLoc == []:    # TO-DO: try to use the numeric answers
            continue
        if lstLoc != [] and type(lstLoc[0]) == str:
            locations = sorted([eval(t) for t in lstLoc])  # from string to 2-tuples
        else:
            locations = lstLoc

        if firstOnly and pos != '0':
            continue

        seq_qid = '%s_%s_%s' % (qid, count, pos)
        with open("%s/%s" % (dirData, row['table_file']), 'rb') as csvfile:
            table_reader = csv.DictReader(csvfile)
            headers = table_reader.fieldnames
            # replace empty column name with _EMPTY_
            headers = map(lambda x: "_EMPTY_" if x == '' else x, headers)
            question = row['question']
            if 'eq_cond' in row:
                eq_cond = row['eq_cond']
            else:
                eq_cond = ''
            eq_cond_column = -1
            eq_cond_column_name = ""
            eq_cond_value = ""
            if eq_cond.find(' = ') != -1:
                f = eq_cond.split(' = ')
                eq_cond_column_name = f[0]
                # replace empty column name with _EMPTY_
                if eq_cond_column_name == '': eq_cond_column_name = "_EMPTY_"
                if len(f) == 2: eq_cond_value = f[1]
                eq_cond_column = headers.index(eq_cond_column_name)

            entries = []
            for row_index, trow in enumerate(table_reader):
                dec_row = []
                for key in table_reader.fieldnames:
                    try:
                        cell = unidecode(trow[key].decode('utf-8')).strip().replace('\n', ' ').replace('"', '')
                    except:
                        cell = ''
                    if cell == '': cell = "_EMPTY_"     # replace empty cell with a special _EMPTY_ token
                    dec_row.append(cell)

                dec_row = dict(zip(headers, dec_row))

                entry = []
                for key in headers:
                    entry.append(dec_row[key])
                entries.append(entry)

        num_cols = len(headers)
        num_rows = len(entries)

        # check annotated table
        tabid = str(row['table_file']).replace('table_csv/', '').replace('.csv','').split('_')
        fnAnnTab = str("%s/%s-annotated/%s.annotated" % (config.d["AnnotatedTableDir"], tabid[0], tabid[1]))
        annTab = loadAnnotatedTable(fnAnnTab, num_rows, num_cols)

        # check which columns are numeric columns
        numeric_cols = set()
        for c in xrange(num_cols):
            sumNumRows = sum([1 for r in xrange(num_rows) if annTab[(r,c)].number != config.NaN or annTab[(r,c)].date != None])
            if sumNumRows >= num_rows-1:    # leave some room for table error
                numeric_cols.add(c)

        types = infer_column_types(headers, entries)
        if locations == []:
            answer_column = -1
            is_col_select = False
            answer_rows = set()
        else:
            columns = [coord[1] for coord in locations]
            col_count = Counter()
            for col in columns:
                col_count[col] += 1
            #print("debug: most_common", col_count.most_common(1), "fnAnnTab", fnAnnTab)
            answer_column = col_count.most_common(1)[0][0]
            if 'complete_match' in row:
                complete_match = row['complete_match']
            else:
                complete_match = ''

			# make sure it's not column selection -- this snippet may have bugs
            is_contig = True
            prev_row_idx = locations[0][0]
            for r, c in locations[1:]:
                if r == prev_row_idx + 1:
                    prev_row_idx = r
                else:
                    is_contig = False
                    break
            is_col_select = False
            if len(locations) == num_rows or (len(locations) == (num_rows - 1) and is_contig):
                is_col_select = True

            answer_rows = set([coord[0] for coord in locations])

        quesInfo = QuestionInfo(seq_qid = seq_qid,      # question id
                                pos = pos,              # the question position in the sequence
                                question = question,                 # question
                                table_file = row['table_file'], # table file name
                                headers = headers,                   # table header fields
                                entries = entries,                   # table content
                                types = types,                       # table column field types
                                answer_column_idx = answer_column,   # answer column index
                                answer_column_name = headers[answer_column],     # answer column field
                                answer_rows = answer_rows,           # answer row indices
                                is_col_select = is_col_select,       # is a column-select-only question?
                                eq_cond_column_idx = eq_cond_column, # column index of Y in (Y=Z condition)
                                eq_cond_column_name = headers[eq_cond_column],   # column index of Y in (Y=Z condition)
                                eq_cond_value = eq_cond_value,       # column value of Z in (Y=Z condition)
                                complete_match = complete_match,     # does our "parse" answer the question?
                                answer_coordinates = locations,         # answer_coordinates
                                answer_text = eval(row['answer_text']), # answer_text
                                annTab = annTab, 
                                numeric_cols = numeric_cols         # set of numeric or date columns
                                )

        data.append(quesInfo)

    return data


# figure out if each column contains text, dates, or numbers
def infer_column_types(headers, entries, min_date=1700, max_date=2016):
    types = []
    for c in range(len(headers)):
        curr_entries = []
        for r in range(len(entries)):
            curr_entries.append(entries[r][c])
        
        curr_type = 'text'

        # check for numbers
        new_entries = []
        for x in curr_entries:
            try:
                # if the first thing in the cell is a number, ignore the rest
                # x = x.replace(',', '')
                y = float(x)
                new_entries.append(y)
            except:
                break

        # now figure out if it's a date or a normal number
        # we'll say dates are everything between 1800-2016
        if len(new_entries) == len(curr_entries):
            curr_type = 'number'

            min_col = min(new_entries)
            max_col = max(new_entries)
            if min_col > min_date and max_col <= max_date:
                curr_type = 'date'

        types.append(curr_type)
    
    return types

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

##------------------------------------------------------------------------------

class AnnotatedTabEntry:
    def __init__(self, row = config.NaN, col = config.NaN, content = "", 
                 strTokens = "", strLemmaTokens = "", strPosTags = "", strNerTags = "", strNerValues = "",
                 number = config.NaN, strDate = "", num2 = config.NaN, 
                 strList = ""):
        self.row = row
        self.col = col
        self.content = content
        self.tokens = self.mysplit(strTokens)
        self.lemmaTokens = self.mysplit(strLemmaTokens)
        self.posTags = self.mysplit(strPosTags)
        self.nerTags = self.mysplit(strNerTags)
        self.nerValues = self.mysplit(strNerValues)
        self.number = number
        if strDate:
            year = 4    # to allow Feb-29
            month = day = 1
            dateFields = self.mysplit(strDate,'-')
            if len(dateFields) == 3:
                if RepresentsInt(dateFields[0]):
                    year = int(dateFields[0])
                    if year > datetime.MAXYEAR: year = datetime.MAXYEAR
                    if year < datetime.MINYEAR: year = datetime.MINYEAR
                if RepresentsInt(dateFields[1]):
                    month = int(dateFields[1])
                if RepresentsInt(dateFields[2]):
                    day = int(dateFields[2])
            #print(strDate, year,month,day)
            self.date = datetime.date(year,month,day)
        else:
            self.date = None
        self.num2 = num2
        self.list = self.mysplit(strList)

    @staticmethod
    def mysplit(field, delim='|'):
        if field: return field.split(delim)
        return []

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def loadAnnotatedTable(fnAnnTab, num_rows, num_cols):
    ret = {}
    with open(fnAnnTab, 'rb') as tsvfile:
        table_reader = csv.DictReader(tsvfile, delimiter='\t')
        for trow in table_reader:
            r = int(trow['row'])
            c = int(trow['col'])
            number = config.NaN
            if trow['number']:
                number = float(trow['number'])
            num2 = config.NaN
            if trow['num2']:
                number = float(trow['num2'])

            entry = AnnotatedTabEntry(r, c, trow['content'],
                                      trow['tokens'], trow['lemmaTokens'], trow['posTags'], trow['nerTags'], trow['nerValues'],
                                      number, trow['date'], num2, trow['list'])
            ret[(r,c)] = entry
    # check if the annotated table is complete
    inComplete = False
    for r in xrange(-1, num_rows):
        for c in xrange(num_cols):
            if (r,c) not in ret:
                ret[(r,c)] = AnnotatedTabEntry()
                inComplete = True
    if inComplete:
        print("Warning: %s is not complete!" % fnAnnTab)

    return ret


reNumber = re.compile(r'\d[\d,\.]*|\.\d[\d,\.]*')
strWdNumbers = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve']
setWdNumbers = set(strWdNumbers)
'''
def findNumbers(strQues):
    ret = set()
    for strNum in reNumber.findall(strQues):
        strN = strNum.replace(',','')
        if is_float_try(strN):
            ret.add(float(strN))
    lowerQ = strQues.lower()
    setWds = set(lowerQ.split(' '))
    for i,w in enumerate(strWdNumbers):
        if w in setWds:
            if i == 1 and "which one" in lowerQ: continue  # ignore "one" in "which one"
            ret.add(float(i))
    return ret
'''
def findNumbers(quesWds):
    ret = []
    for i,tok in enumerate(quesWds):
        m = reNumber.match(tok)
        if not m: continue
        strN = m.group().replace(',','')
        if is_float_try(strN):
            ret.append((i,float(strN)))
        
        if tok in setWdNumbers:
            ret.append((i, float(strWdNumbers.index(tok))))

    #print("len(ret) = ", len(ret))
    return ret
        