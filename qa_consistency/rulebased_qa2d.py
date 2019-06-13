#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import sys
import os
import re
# sys.path.append('/home/marcotcr/pythonlibs/pattern')
from copy import deepcopy
import codecs
from mosestokenizer import MosesDetokenizer
from conllu import parse
import pattern
import pattern.en
import spacy
from allennlp.predictors.predictor import Predictor
import numpy as np
from .question_repr import QuestionRepr

# this hack is required because qanli has a file hardcoded in it (preps.txt)
current_path = os.getcwd()
this_dir, _ = os.path.split(__file__)
os.chdir(this_dir)
from qanli.rule import Question, AnswerSpan
os.chdir(current_path)


def matchz(span, match_word):
    if match_word in span._.labels:
        return True
    if type(match_word) != str:
        if np.any([x in span._.labels for x in match_word]):
            return True
    if len(span) != 1:
        return False
    if type(match_word) == str:
        return span[0].tag_ == match_word or span[0].pos_ == match_word or span[0].text == match_word or span[0].lemma_ == match_word
    else:
        return span[0].tag_ in match_word or span[0].pos_ in match_word or span[0].text in match_word or span[0].lemma_ in match_word

def find_sequence(doc, seq, verbose=False, start=None, end=None):
    if type(doc) == spacy.tokens.doc.Doc:
        sent = list(doc.sents)[0]
    else:
        sent = doc
    consts = list(sent._.constituents)
    if verbose:
        print(sent._.parse_string)
    start = sent.start if not start else start
    end = sent.end if not end else end
    if seq[0] == '^':
        current = [[(start, start)]]
    else:
        current = [[(i.start, i.end)] for i in consts if matchz(i, seq[0]) and i.start >= start and i.end <= end]
    until_end = False
    if seq[-1] == '$':
        seq = seq[:-1]
        until_end = True
    for s in seq[1:]:
        new_current = []
        for c in current:
            next_ = [(i.start, i.end) for i in consts if matchz(i, s) and i.start == c[-1][-1] and i.end <= end]
            if next_:
                for n in next_:
                    new_current.append(c + [n])
        current = new_current
    if seq[0] == '^':
        current = [x[1:] for x in current]
    if until_end:
        current = [x for x in current if x[-1][1] == end]
    return current

def sequence_filter(seq_sequence, parsed_questions, ans_sequence, parsed_answers):
    # seq_sequence is a list of sequences to be progressively applied. For example:
    # [['SQ'], ['NP', 'VP']] will find sequences with an SQ, then sequences where inside the SQ there is an NP, VP
    # Returns tuple (ids, spans)
    found = []
    if seq_sequence:
        seq = seq_sequence[0]
        found = [(i, find_sequence(d, seq)) for (i, d) in enumerate(parsed_questions) if d]
        found = [x for x in found if x[1]]
    else:
        found = [(i, []) for i in range(len(parsed_questions))]
    for seq in seq_sequence[1:]:
        new_found = []
        for idx, span in found:
            span = span[0]
            span_range = (span[0][0], span[-1][1])
            span = parsed_questions[idx][slice(*span_range)]
            f = find_sequence(span, seq)
            if f:
                new_found.append((idx, f))
        found = new_found
    for seq in ans_sequence:
        new_found = []
        for idx, span in found:
            f = find_sequence(parsed_answers[idx], seq)
            if f:
                new_found.append((idx, span))
        found = new_found
    if not found:
        return ([], [])
    idxs, spans = map(list, list(zip(*found)))
    return (idxs, spans)


def map_dep(dep):
    if dep == 'nsubjpass':
        return 'nsubj:pass'
    if dep == 'auxpass':
        return 'aux:pass'
    if dep == 'compoundprt':
        return 'compound:prt'
    return dep

def clean_answer(ans):
    return re.sub(r'^(the|a|an) ', '', ans)

class RuleBasedQa2D:
    def __init__(self, dependency, nlp):
        self.detokenizer = MosesDetokenizer('en')
        self.dependency = dependency
        self.nlp = nlp

    def our_rules(self, parsed_qa):
        seq_sequence = [['^', ['WHNP', 'WHADVP'], 'S', '?']]
        sliced, spans = sequence_filter(seq_sequence, [parsed_qa.question_const_parse], [], [parsed_qa.answer_const_parse])
        if sliced:
            imp = self.whnp_s(parsed_qa, spans[0])
            if imp:
                return imp
        seq_sequence = [['SQ'], ['^', 'VERB', 'NP', '$']]
        sliced, spans = sequence_filter(seq_sequence, [parsed_qa.question_const_parse], [], [parsed_qa.answer_const_parse])
        if sliced:
            imp = self.sq_verb_np(parsed_qa, spans[0])
            if imp:
                return imp
        return ''

    def qa2d(self, parsed_qa):
        ours = self.our_rules(parsed_qa)
        if ours:
            return ours
        doc = parsed_qa.question_const_parse
        ans = parsed_qa.answer_const_parse
        q = Question(deepcopy(parse(self.to_connl(doc, parsed_qa.question_dep_parse))[0]))
        if not q.isvalid:
    #         print("Question is not valid.".format(idx))
            return ''
        a = AnswerSpan(deepcopy(parse(self.to_connl(ans, None))[0]))
        if not a.isvalid:
    #         print("Answer span is not valid.".format(idx))
            return ''
        try:
            q.insert_answer_default(a)
        except:
            print('Error in ', parsed_qa)
            return ''
        return self.detokenizer(q.format_declr())
    #     return detokenizer.detokenize(q.format_declr(), return_str=True)

    def to_connl(self, const_parse, dep_parse=None):
        ret = []

        if dep_parse is None:
            dep_pred = self.dependency.predict(sentence=const_parse.text)
        else:
            dep_pred=  dep_parse
    #         print(dep_pred['predicted_dependencies'])
        if len(dep_pred['words']) != len(const_parse):
            const_parse = [x for x in const_parse if x.text in dep_pred['words']]
        for i, word in enumerate(const_parse):
    #         print(i)
            head = word.head.i if dep_pred is None else dep_pred['predicted_heads'][i]
            if head == i:
                head_idx = 0
            else:
                if dep_pred:
                    head_idx = head
                else:
                    head_idx = head - 1
            ret.append("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(
                     i+1, # There's a word.i attr that's position in *doc*
                      word,
                '_', #word.lemma_,
                      word.pos_, # Coarse-grained tag
                      word.tag_ if dep_pred is None else dep_pred['pos'][i], # Fine-grained tag
                      '_',
                      str(head_idx),
                      word.dep_.lower() if dep_pred is None else map_dep(dep_pred['predicted_dependencies'][i]), # Relation
                    '_',
                    '_'))
        return '\n'.join(ret)

    def whnp_s(self, parsed_qa, span):
        q = QuestionRepr(parsed_qa)
        qtext = parsed_qa.question
        # parts = get_parts((q, span))
        # print(parts)
        # new_q = '%s %s %s' % (answer, parts[0][2:], parts[1])
        to_remove = set([x['id'] for x in q.question if x['word'] in ['Where', '?', 'What', 'Who', 'When']])
        subjs = q.get_children(q.root, ['nsubj', 'csubj', 'nsubjpass'], 'anywhere')
        panswer = parsed_qa.answer_const_parse
        answer = parsed_qa.answer
        passage = parsed_qa.context
        if passage is not None:
            for start in [m.start() for m in re.finditer(re.escape(answer), passage)]:
                if start == 0:
                    continue
                prev = passage[:start].split()[-1]
                # print(prev)
                if prev.lower() in ['the', 'a', 'an']:
                    answer = '%s %s' % (prev, answer)
                    break
        # subjs = [x for x in subjs if 0 in [y['id'] for y in q.get_subtree(x)]]
        # print(subjs)
        preps = []
        subj_ids = []
        if subjs:
            # subj_ids = sorted([x['id'] for y in subjs for x in q.get_subtree(y)])
            subj_ids = sorted([x['id'] for x in q.get_subtree(subjs[0])])
            preps = q.get_children(subjs[0], 'prep')
        else:
            subj_ids = range(*span[0][0])
        # if not set(list(range(*span[0][0]))).intersection(set(subj_ids)):
        # if span[0][0][0] not in subj_ids:
            # print(span)
            # print(set(list(range(*span[0][0]))))
            # print(set(subj_ids))
            # print(question['const_parse'], answer)
        subj_preps = set()
        for p in preps:
            # print(qtext)
            subj_preps = subj_preps.union(set([x['id'] for x in q.get_subtree(p)]))
        prap = ' '.join([q.question[x]['word'] for x in subj_preps])
        # print(prap)

        # if qtext.startswith('How many'):
        #     to_remove = to_remove.union(set([0, 1]))
        subj_lemmas = [q.question[i]['lemma'] for i in subj_ids]
        # if q.question[subj_ids[0]]['lemma'] == 'how' and q.question[subj_ids[1]]['lemma'] == 'many':
        #     to_remove = to_remove.union(set(subj_ids[:2]))
        types = [(x, y) for x in ['what', 'which'] for y in ['kind', 'type', 'sort']]
        types.append(('how', 'many'))
        if tuple(subj_lemmas[:2]) in set(types):
            if tuple(subj_lemmas[:2]) == ('how', 'many'):
                to_insert = [x for x in subj_ids[2:]]
            else:
                to_insert = [x for x in subj_ids[3:]]
            lemmas = set([q.question[i]['lemma'] for i in to_insert])
            alemmas = set([a.lemma_ for a in panswer])
            if lemmas.intersection(alemmas):
                to_insert = []
            to_remove = to_remove.union(set([x for x in subj_ids if x not in to_insert]))
        else:
            to_remove = to_remove.union(set([x for x in subj_ids if x not in subj_preps]))

        bef = [x['word'] for x in q.question if x['id'] not in to_remove and x['id'] < subj_ids[0]]
        rest = [x['word'] for x in q.question if x['id'] not in to_remove and x['id'] >= subj_ids[0]]
        new_q = ' '.join(bef + [answer.strip('.')] + rest + ['.'])
        new_q = new_q[0].upper() + new_q[1:]
        # new_q = '%s %s .' % (answer.strip('.'), ' '.join(rest))
        return new_q

    def sq_verb_np(self, parsed_qa, span):
        q = QuestionRepr(parsed_qa)
        qtext = parsed_qa.question
        panswer = parsed_qa.answer_const_parse
        answer = parsed_qa.answer
        passage = parsed_qa.context
        to_remove = set([x['id'] for x in q.question if x['word'] in ['Where', '?', 'What', 'Who', 'When']])
        # subjs = q.get_
        subjs = q.get_children(q.root, ['nsubj', 'csubj', 'nsubjpass'], 'anywhere')
        # subjs = [x for y in ['nsubj', 'csubj', 'nsubjpass'] for x in q.get_all(y)]
        tobe = q.question[span[0][0][0]]['lemma'] == 'be'
        if passage is not None:
            for start in [m.start() for m in re.finditer(re.escape(answer), passage)]:
                if start == 0:
                    continue
                prev = passage[:start].split()[-1]
                # print(prev)
                if prev.lower() in ['the', 'a', 'an']:
                    answer = '%s %s' % (prev.lower(), answer)
                    break
        preps = []
        subj_ids = []
        # if any([qtext.startswith(x) for x in ['What', 'How many', 'Who']]):
        #     return []
        if subjs:
            subj_ids = sorted([x['id'] for x in q.get_subtree(subjs[0])])
            preps = q.get_children(subjs[0], 'prep')
            end = span[0][0][0]
            wh = find_sequence(parsed_qa.question_const_parse, [['WHNP', 'WHADVP', 'WHADJP', 'WHPP']], end=end)
            bla = list(range(*wh[-1][0]))
            # bla = list(range(0, end))
            if len(set(subj_ids).intersection(set(bla))) == len(bla):
                pass
            else:
                subj_ids = bla
            subj_ids = [x for x in subj_ids if x < end]
            # print(subj_ids, bla, list(range(0, end)))
            # return []
        else:
            return ''
            end = span[0][0][0]
            # wh = find_sequence(question['const_parse'], [['WHNP', 'WHADVP', 'WHADJP', 'WHPP']], end=end)
            # wh = sorted(wh, key=lambda x:x[0][1], reverse=True)
            # subj_ids = range(*wh[0][0])
        subj_preps = set()
        for p in preps:
            # print(qtext)
            subj_preps = subj_preps.union(set([x['id'] for x in q.get_subtree(p)]))
        prap = ' '.join([q.question[x]['word'] for x in subj_preps])
        subj_lemmas = [q.question[i]['lemma'] for i in subj_ids]
        types = [(x, y) for x in ['what', 'which'] for y in ['kind', 'type', 'sort']]
        types.append(('how', 'many'))
        #     to_remove = to_remove.union(set(subj_ids[:2]))
        if tuple(subj_lemmas[:2]) in set(types):
            if tuple(subj_lemmas[:2]) == ('how', 'many'):
                to_insert = [x for x in subj_ids[2:]]
            else:
                to_insert = [x for x in subj_ids[3:]]
            lemmas = set([q.question[i]['lemma'] for i in to_insert])
            alemmas = set([a.lemma_ for a in panswer])
            if lemmas.intersection(alemmas):
                to_insert = []
            to_remove = to_remove.union(set([x for x in subj_ids if x not in to_insert]))
        else:
            to_remove = to_remove.union(set([x for x in subj_ids if x not in subj_preps]))

        if tobe:
            to_move = []
            if q.root['lemma'] == 'be':
                to_move.append(q.root)
            to_move.extend([x for x in q.get_children(q.root, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')])
            if q.question[span[0][0][0]]['lemma'] == 'be' and q.question[span[0][0][0]] not in to_move:
                to_move.append(q.question[span[0][0][0]])
            to_move = [x for x in to_move if x['lemma'] in ['be', 'can', 'do', 'will']]
            others = [x for x in subj_ids if x not in to_remove]
            # TODO: used to be just aux, check if this is correct
            # if any([x['dep'] in ['aux', 'cop'] for x in to_move]):
            #     to_move = [x for x in to_move if x['dep'] not in ['auxpass', 'advmod']]
            if any([x['lemma'] in ['be'] for x in to_move]):
                to_move = sorted([x for x in to_move if x['lemma'] in ['be']], key=lambda z:z['id'])[:1]
            to_remove = to_remove.union(set([x['id'] for x in to_move])).union(set(others))
            wto_move = [x['word'] for x in to_move]
            bef = [x for x in q.question if x['id'] not in to_remove and x['id'] < subj_ids[0]]
            if len(bef) == 1 and bef[0]['pos'] == 'ADP':
                wto_move = wto_move + [bef[0]['word'].lower()]
                bef = []
            else:
                bef = [x['word'] for x in bef]
            wothers = [x['word'] for x in q.question if x['id'] in others]
            rest = [x['word'] for x in q.question if x['id'] not in to_remove and x['id'] >= subj_ids[0]]
            new_q = ' '.join(bef + rest + wto_move + [answer.strip('.')] + wothers +  ['.'])

        else:
            bef = [x['word'] for x in q.question if x['id'] not in to_remove and x['id'] < subj_ids[0]]
            rest = [x['word'] for x in q.question if x['id'] not in to_remove and x['id'] >= subj_ids[0]]
            new_q = ' '.join(bef + [answer.strip('.')] + rest + ['.'])
        new_q = new_q[0].upper() + new_q[1:]
        return new_q
