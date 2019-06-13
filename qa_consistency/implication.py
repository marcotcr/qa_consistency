#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import spacy
from benepar.spacy_plugin import BeneparComponent
from allennlp.predictors.predictor import Predictor
import pattern
import pattern.en
import re
from . import rulebased_qa2d
from . import language_model
from .question_repr import QuestionRepr
import tqdm
from pattern.en import wordnet, pluralize
import os

def change_number(verb, plural=True):
    params = pattern.en.tenses(verb)
    if not params:
        return verb
    params = list(params[-1])
    params[2] = 'plural' if plural else 'singular'
    return pattern.en.conjugate(verb, *params)

def capitalize(x):
    if not x:
        return x
    return x[0].capitalize() + x[1:]

def insert_articles_in_answer(answer):
    remove_empty = lambda x: filter(lambda y: y != '', x)
    articles = ['', 'a', 'the', 'an']
    anss = answer.split()
    ret = [answer]
    for i in range(1, len(anss)):
        before = anss[:i]
        after = anss[i:]
        ret.extend([' '.join(remove_empty(before + [x] + after)) for x in articles])
    return ret

def conjugate_x_like_y(x, y):
    p = pattern.en.tenses(y)
    params = []
    if p:
        params = list(p[0])
        return pattern.en.conjugate(x, *params)
    else:
        return x


def get_antonyms(word, pos=None):
    map_pos = {'VERB': wordnet.VERB, 'ADJ': wordnet.ADJECTIVE, 'NOUN': wordnet.NOUN, 'ADV': wordnet.ADVERB}
    pos = map_pos[pos]
    synonyms = []
#     antonyms = collections.defaultdict(lambda: [])
    antonyms = []
    for syn in wordnet.synsets(word, pos=pos):
        # if syn.senses[0] != word:
        #     continue
        if syn.antonym:
            # print(syn, syn.antonym)
            for x in syn.antonym:
                antonyms.append((x[0], 1))
                # antonyms.extend([(a, 1) for a in x.senses])
    return antonyms

def get_related(word, plural=False):
    syn =  wordnet.synsets(word)
    if not syn:
        return []
    syn = syn[0]
    if not syn.hypernym:
        return []
    fn = pluralize if plural else lambda x:x
    s = sorted([(fn(' '.join(x[0].split('_'))), x.similarity(syn)) for x in syn.hypernym.hyponyms()], key=lambda z:z[1], reverse=True)
    return [x for x in s if x[1] > 0.5 and x[1] < 1 and word not in x[0]]

def who_or_what(spacy_span):
    return 'Who' if (
        spacy_span[-1].ent_type_ == 'PERSON' or
        (len(spacy_span) == 1 and
          (spacy_span.text[0].isupper() and spacy_span[0].i != 0) or spacy_span[0].pos_ =='PROPN'
        )) else 'What'

def color_opposite(color):
    opposites = {'black': 'white',
                 'white': 'black',
                 'green': 'red',
                 'red': 'green',
                 'blue': 'orange',
                 'orange': 'blue',
                 'yellow': 'violet',
                 'violet': 'yellow',
                 'pink': 'green',
                'multicolored': 'black and white',
                'black and white': 'multicolored',
                'brown': 'pink',
                'color': 'black and white',
                'multi': 'black and white',
                'gray': 'red',
                'purple': 'green',
                }
    colors_mentioned = [x for x in color.split() if x in opposites]
    opp = [opposites[x] for x in colors_mentioned if x in opposites and opposites[x] not in colors_mentioned]
    if not opp:
        opposite_order = ['black', 'white', 'green', 'red', 'blue', 'orange', 'yellow', 'pink']
        return [x for x in opposite_order if x not in colors_mentioned][0]
    return opp[0]


def clean_answer(ans):
    return re.sub(r'^(the|a|an) ', '', ans)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def batch_predict(predict_fn, questions, batch_size=100, verbose=True):
    out = []
    total = len(questions) / batch_size
    cs = chunks(questions, batch_size)
    if verbose:
        cs = tqdm.tqdm(cs, total=total)
    for c in cs:
        out.extend(predict_fn([{'sentence': x} for x in c]))
    return out


def find_seq(spacy_doc, seq):
    # first ocu
    words = [x.text for x in spacy_doc]
    pos = [x.pos_ for x in spacy_doc]
    tag = [x.tag_ for x in spacy_doc]
    ret = []
    for i, x in enumerate(spacy_doc):
        if seq[0] in [words[i], pos[i], tag[i]]:
            failed = False
            for j, y in enumerate(seq[1:], start=i + 1):
                if y not in [words[j], pos[j], tag[j]]:
                    failed = True
                    break
            if not failed:
                ret.append(i)
    return ret

class ParsedQA:
    def __init__(self, question_const_parse, question_dep_parse, answer_const_parse, context=None):
        self.question_const_parse = question_const_parse
        self.question_dep_parse = question_dep_parse
        self.answer_const_parse = answer_const_parse
        self.question = question_const_parse.text
        self.answer = answer_const_parse.text
        self.context = context

    def __repr__(self):
        if hasattr(self, 'dec_const_parse'):
            return 'Q: %s\nA: %s\nD: %s' %  (self.question, self.answer, self.dec_const_parse)
        return 'Q: %s\nA: %s' %  (self.question, self.answer)

    def as_tuple(self):
        if self.context is not None:
            return (self.question, self.answer, self.context)
        else:
            return (self.question, self.answer)

class QAParser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(BeneparComponent("benepar_en_small"))
        self.dependency = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
        self.rule_qa2d = rulebased_qa2d.RuleBasedQa2D(self.dependency, self.nlp)
    def parse_qa(self, question, answer, context=None, add_declarative=False):
        return self.parse_qas([question], [answer], [context], add_declarative=add_declarative)[0]

    def parse_qas(self, questions, answers, contexts=None, verbose=False, add_declarative=False):
        unique_questions = list(set(questions))
        if '' in unique_questions:
            unique_questions.remove('')
        unique_qpos = dict([(x, i) for i, x in enumerate(unique_questions)])
        unique_qpos[''] = len(unique_qpos)
        unique_answers = list(set(answers))
        unique_apos = dict([(x, i) for i, x in enumerate(unique_answers)])
        to_parse = self.nlp.pipe(unique_questions)
        if verbose:
            print('Const parse questions')
            to_parse = tqdm.tqdm(to_parse, total=len(unique_questions), miniters=1000)
        const = list(to_parse)
        const.append(self.nlp(''))
        ans = list(self.nlp.pipe(unique_answers))
        if verbose:
            print('Dep parse questions')
        dep = batch_predict(self.dependency.predict_batch_json, unique_questions, verbose=verbose)
        dep.append({'words': [], 'pos': [], 'predicted_dependencies': [], 'predicted_heads': []})
        if contexts is None:
            contexts = [None for _ in range(len(questions))]
        ret = [ParsedQA(const[unique_qpos[q]], dep[unique_qpos[q]], ans[unique_apos[a]], c) for q, a, c in zip(questions, answers, contexts)]
        if add_declarative:
            self.add_declarative(ret)
        return ret

    def add_declarative(self, parsed_qas):
        decs = [self.rule_qa2d.qa2d(q) for q in parsed_qas]
        parsed = self.parse_qas(decs, ['' for x in decs])
        for x, p in zip(parsed_qas, parsed):
            x.dec_const_parse = p.question_const_parse
            x.dec_dep_parse = p.question_dep_parse
    def qa2d(self, question, answer):
        return self.rule_qa2d.qa2d(question, answer)

class ImplicationsSquad:
    def __init__(self, qa_parser=None):
        if qa_parser is not None:
            self.parser = qa_parser
        else:
            self.parser = QAParser()
        self.nlp = self.parser.nlp
        this_dir, _ = os.path.split(__file__)
        lm_path = os.path.join(this_dir, 'question_4gram.arpa')
        self.lm = language_model.LanguageModel(lm_path)
        self.score_product = self.lm.score_product
        self.get_best = lambda x:self.score_product(x)[0][0]
        pass

    def parse_dataset(self, questions, answers, passages, **kwargs):
        return self.parser.parse_qas(questions, answers, passages, add_declarative=True, **kwargs)

    def implications(self, question, answer, context):
        parsed_qa = self.parser.parse_qa(question, answer, context, add_declarative=True)
        return self.implications_from_parsed(parsed_qa)

    def implications_from_parsed(self, parsed_qa):
        try:
            imps = self.declarative_to_q(parsed_qa)
        except:
            print('Error', parsed_qa)
            return []
        imps = [(x[0], clean_answer(x[1]), x[2]) for x in imps]
        imps = [x for x in imps if x[1] in parsed_qa.context and x[1] != parsed_qa.answer and x[1]]
        return imps

    def declarative_to_q(self, parsed_qa):
        # obj must have dep_parse, const_parse and passage
        imp = []
        imp += self.sentence_to_preps(parsed_qa)
        imp += self.sentence_to_which(parsed_qa)
        imp += self.sentence_to_subj_obj(parsed_qa)
        return imp
    def wh_picker(self, parsed_qa, answer_idxs):
        q = parsed_qa.dec_const_parse
        qu = QuestionRepr(parsed_qa, use_declarative=True)
        # print(parsed_qa)
        # print(q[answer_idxs[0]:answer_idxs[-1]+1])
        if q[0].pos_ != 'PROPN':
            qu.question[0]['word'] = qu.question[0]['word'].lower()
        dobj = qu.get_children(qu.root, 'dobj')
        subj = qu.get_children(qu.root, ['nsubj', 'csubj', 'nsubjpass'], 'anywhere')
        if subj and set(answer_idxs) == set([x['id'] for x in qu.get_subtree(subj[0])]):
            if dobj:
                dobj = qu.get_subtree(dobj[0])
                if any([x['word'] in ['his', 'her', 'he', 'she', 'him', 'her'] for x in dobj]):
                    return 'Who'
                if any([x['word'] in ['its', 'it'] for x in dobj]):
                    return 'What'
            if all([qu.question[i]['word'][0].isupper() for i in answer_idxs]) and qu.root['pos'] == 'VERB' and qu.root['lemma'] not in ['cause', 'be', 'take', 'have', 'can', 'use'] and not qu.get_children(qu.root, ['aux', 'auxpass', 'cop', 'dep'], 'anywhere'):
                return 'Who'

        ents = [q[i].ent_type_ for i in answer_idxs if q[i].pos_ not in ['CCONJ', 'ADP', 'DET', 'PUNCT', 'PART']]
        prep = ''
        if q[answer_idxs[0]].pos_ == 'ADP' or (answer_idxs[0] != 0 and q[answer_idxs[0] - 1].pos_ == 'ADP'):
            prep = q[answer_idxs[0]] if q[answer_idxs[0]].pos_ == 'ADP' else q[answer_idxs[0] - 1]
            prep = prep.text
        what = set(['EVENT'])
        qwords = [a.text for a in q]
        # if any([x in ['his', 'her', 'he', 'she'] for x in qwords]):
        #     return 'Who'
        if any([x in what for x in ents]) and not prep in ['at', 'in']:
            return 'What'
        who = set(['PERSON'])
        if all([x in who for x in ents]) and len(ents) > 1 and not prep in ['at', 'in', 'since']:
            return 'Who'
        if any([x in who for x in ents]) and not prep in ['at', 'in', 'since', 'of']:
            return 'Who'
        if any([x in ['GPE'] for x in ents]) and prep in ['with']:
            return 'Who'
        where = set(['GPE', 'LOC', 'FAC'])
        if all([x in where for x in ents]) and not prep in ['for']:
            return 'Where'
        when = set(['DATE', 'TIME', 'CARDINAL'])
        if all([x in when for x in ents]):
            # if not all([x in when for x in ents]):
            #     print(q)
            #     print(' '.join([q[i].text for i in answer_idxs]))
            #     print(ents)
            return 'When'
        if prep in ['at', 'in']:
            return 'Where'
        if prep in ['since', 'after']:
            return 'When'
        if prep in ['because']:
            return 'Why'
        return 'What'
    def filter_imp(self, parsed_qa):
        q = QuestionRepr(parsed_qa, use_declarative=True)
        forbidden_lemmas = set(['do', 'be', 'have', 'will', 'must', 'can'])
        if q.question[1]['lemma'] in ['do']:
            return True
        if q.question[0]['lemma'] in forbidden_lemmas:
            return True
        if q.question[0]['dep'] in ['aux', 'auxpass', 'cop', 'dep']:
            return True
        return False
    def filter_do(self, parsed_qa):
        q = QuestionRepr(parsed_qa, use_declarative=True)
        forbidden_lemmas = set(['do'])
        if q.question[1]['lemma'] in ['do']:
            return True
        if q.question[0]['lemma'] in forbidden_lemmas:
            return True
        return False
    def sentence_to_which(self, parsed_qa):
        if not len(parsed_qa.dec_const_parse):
            return []
        q = QuestionRepr(parsed_qa, use_declarative=True)
        if self.filter_imp(parsed_qa):
            return []
        qu = parsed_qa.dec_const_parse
        ret = []
        amods = sorted([sorted(q.get_subtree(amod), key=lambda x:x['id']) for amod in q.get_all('amod')], key=lambda a:a[0]['id'])
        to_remove = set([x['id'] for x in q.question if x['word'] in ['..', '.', '?']])
        # join contiguous amods
        for i in range(len(amods) - 1, 0, -1):
            if amods[i][0]['id'] == amods[i - 1][-1]['id'] + 1:
                amods[i - 1].extend(amods[i])
                amods.pop(i)
        for amod in amods:
            which = 'which'
            # a = sorted(q.get_subtree(amod), key=lambda x:x['id'])
            a_words = [x['word'] for x in amod]
            a = [x['id'] for x in amod]
            prev = None if a[0] == 0 else q.question[a[0] - 1]
            next = None if a[-1] >= len(q.question) -1 else q.question[a[-1] + 1]
            rem = to_remove.union(set(a))
            if next and next['pos'] != 'NOUN':
                continue
            if amod[0]['tag'] in ['JJR', 'JJS', 'RBS', 'RBR'] or (prev and prev['tag'] == 'POS'):
                continue
            if prev and (prev['pos'] in ['NOUN', 'CD', 'PUNCT', 'NUM'] or prev['lemma'] in ['which'] or prev['tag'] in ['NNP', 'NNPS']):
                continue
            # TODO these ones work well, no need to uncomment
            # if prev is None:
            #     continue
            # if prev and prev['pos'] == 'ADP':
            #     continue
            if prev and prev['lemma'] == 'the':
                rem.add(prev['id'])
            if prev and prev['lemma'] in ['a', 'an']:
                rem.add(prev['id'])
                which = 'which kind of'
            if prev and prev['lemma'] in ['be']:
                which = 'which kind of'

            # else:
            #     continue

            # if q.question[a[0]]['pos'] == 'ADJ':
            #     continue

            bef = [x['word'] for x in q.question if x['id'] not in rem and x['id'] <= a[0]]
            after = [x['word'] for x in q.question if x['id'] not in rem and x['id'] >= a[0]]
            if not bef:
                which = which.capitalize()
            if self.is_answer_allowed([q.question[i] for i in a]):
                new_q = ' '.join(bef + [which] + after + ['?'])
                # if len(bef) and (len(bef) + len(after)) > 15:
                #     continue
                ret.append((new_q, ' '.join(a_words), 'amod'))
        return ret

    def sentence_to_preps(self, parsed_qa):
        if not len(parsed_qa.dec_const_parse):
            return []
        q = QuestionRepr(parsed_qa, use_declarative=True)
        to_remove = set([x['id'] for x in q.question if x['word'] in ['.', '?']])
        ret = []
        qu = parsed_qa.dec_const_parse
        preps = q.get_all('prep')
        # allz = ['without', 'from', 'in', 'to', 'under', 'upon', '1', 'by', 'toward', 'outside', 'unto', 'for', 'at', 'during', 'above', 'after', 'than', 'since', 'across', 'beyond', 'instead', 'before', 'like', 'around', '(', 'between', 'on', 'according', 'towards', 'near', 'v', 'such', 'within', 'due', 'with', 'apart', 'ABC', 'against', 'per', 'of', 'through', 'below', 'cause', 'over', 'along', 'as', 'forth', 'among', 'following', 'including', 'into', 'out', 'about', 'because', 'inside', 'throughout', 'alongside', 'up', 'When']
        exclude = ['without', 'by', 'toward', 'during', 'above', 'than', 'before', 'between', 'within', 'of', 'below', 'over', 'along', 'following', 'under']
        include = ['in', 'at', 'because', 'near']
        # include_and_change = ['after', 'on']
        include_and_change = []
        exclude_and_change = ['for', 'upon', 'by', 'since', 'with', 'through', 'as', 'among', 'into', 'about', 'from', 'to', 'by', 'after', 'on']
        # in -> where
        # at -> where
        # since -> when
        # because -> why
        for p in preps:
            q = QuestionRepr(parsed_qa, use_declarative=True)
            prep = sorted([x['id'] for x in q.get_subtree(p)])
            prep_words = [q.question[i]['word'] for i in prep]
            if q.question[p['head']]['dep'] in ['nsubj', 'csubj', 'nsubjpass'] and prep_words[0] not in exclude:
                continue
            # if prep_words[0] in include_and_change:
            #     print(qu)
            #     print(' '.join(prep_words))
            #     print()
            if prep_words[0] in exclude:
                if len(prep) == 1:
                    continue
                if self.filter_imp(parsed_qa):
                    continue
                prep = prep[1:]
                prep_words = [q.question[i]['word'] for i in prep]
                wh_word = self.wh_picker(parsed_qa, prep)
                rem = to_remove.union(prep)
                bef = [x['word'] for x in q.question if x['id'] not in rem and x['id'] <= prep[0]]
                after = [x['word'] for x in q.question if x['id'] not in rem and x['id'] >= prep[0]]
                if self.is_answer_allowed([q.question[i] for i in prep]):
                    ret.append((' '.join(bef + [wh_word.lower()] + after + ['?']), ' '.join(prep_words), wh_word.lower()))
            elif prep_words[0] in include + include_and_change + exclude_and_change:
                wh_word = self.wh_picker(parsed_qa, prep)
                if prep_words[0] in include_and_change + exclude_and_change:
                    rem = to_remove.union(prep[1:])
                    prep_words = prep_words[1:]
                else:
                    rem = to_remove.union(prep)
                    prep_words = prep_words[1:]
                if qu[0].tag_ not in ['NNP', 'NNPS']:
                # if qu[0].ent_type_ == '':
                    q.question[0]['word'] = q.question[0]['word'].lower()

                to_move = []
                if q.root['lemma'] == 'be':
                    to_move.append(q.root)
                to_move.extend([x for x in q.get_children(q.root, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')])
                to_move = [x for x in to_move if x['lemma'] in ['be', 'can', 'do', 'will', 'would', 'have', 'may']]
                if any([x['dep'] == 'aux' for x in to_move]):
                    to_move = [x for x in to_move if x['dep'] not in ['auxpass', 'advmod', 'root']]
                if any([x['lemma'] in ['be'] for x in to_move]):
                    to_move = sorted([x for x in to_move if x['lemma'] in ['be']], key=lambda z:z['id'])[:1]
                if to_move:
                    # do = ' '.join([x['word'] for x in to_move])
                    rem = rem.union(set([x['id'] for x in to_move]))
                # else:
                #     q.root['word'] = pattern.en.conjugate(q.root['word'], 'infinitive')


                add_does = q.root['pos'] in ['VERB'] and q.root['lemma'] not in ['be', 'do'] and not to_move
                # print('\n'.join(map(str, q.question)))
                do = []
                rem = rem.union([x['id'] for x in to_move])
                if add_does:
                    p = pattern.en.tenses(q.root['word'])
                    tense = 'past' if self.nlp.vocab.morphology.tag_map[q.root['tag']].get('Tense') == 'past' else 'present'
                    params = [tense, 3]
                    if p:
                        params = list(p[0])
                    # print(prep_words)
                    # print(params)
                    # print(q.root['word'])
                    # print()
                    do = [pattern.en.conjugate('do', *params)]
                    q.root['word'] = pattern.en.conjugate(q.root['word'], 'infinitive')
                advcl = q.get_children(q.root, ['advcl'], 'left')
                before_wh = set()
                for a in advcl:
                    a = q.get_subtree(a)
                    before_wh = before_wh.union([x['id'] for x in a])
                    if before_wh.intersection(rem):
                        return ret
                    rem = rem.union([x['id'] for x in a])
                if before_wh:
                    wh_word = wh_word.lower()
                    next = q.question[sorted(before_wh)[-1]]['id'] + 1
                    if next < len(q.question) and q.question[next]['pos'] == 'PUNCT':
                        before_wh.add(next)
                        rem.add(next)
                bef = [q.question[i]['word'] for i in sorted(before_wh)] + [wh_word] + do + [x['word'] for x in sorted(to_move, key=lambda a: a['id'])]
                after = [x['word'] for x in q.question if x['id'] not in rem]
                if self.is_answer_allowed([q.question[i] for i in prep[1:]]):
                    ret.append((' '.join(bef + after + ['?']), ' '.join(prep_words), wh_word.lower()))
        return ret

    def is_answer_allowed(self, answer_list):
        if len(answer_list) == 0:
            return False
        if len(answer_list) == 1 and answer_list[0]['pos'] in ['PRON', 'DET', 'ADP']:
            return False
        if len(answer_list) == 1 and len(answer_list[0]['word']) < 3:
            return False

        # print(answer_list, answer_list[0]['pos'])
        # print([x['word'] for x in answer_list])
        return True

    def sentence_to_subj_obj(self, parsed_qa):
        if not len(parsed_qa.dec_const_parse):
            return []
        q = QuestionRepr(parsed_qa, use_declarative=True)
        qu = parsed_qa.dec_const_parse
        dobj = q.get_children(q.root, 'dobj')
        subj = q.get_children(q.root, ['nsubj', 'csubj', 'nsubjpass'], 'anywhere')
        to_remove = set([x['id'] for x in q.question if x['pos'] == 'PUNCT'])
        ret = []
        if qu[0].pos_ != 'PROPN':
            q.question[0]['word'] = q.question[0]['word'].lower()
        if subj:
            sub = sorted([x for y in subj for x in q.get_subtree(y)], key=lambda a:a['id'])
            sub_ids = [x['id'] for x in sub]
            spacy_span = parsed_qa.question_const_parse[min(sub_ids): max(sub_ids) + 1]
            # start = who_or_what(spacy_span)
            start = self.wh_picker(parsed_qa, sub_ids)
            rem = to_remove.union(set(sub_ids))
            saved = q.root['word']
            if q.root['pos'] == 'VERB':
                p = pattern.en.tenses(q.root['word'])
                if p:
                    auxs = [x for x in q.get_children(q.root, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')]
                    if not auxs and q.root['lemma'] != 'do':
                        params = list(p[-1])
                        params[1] = 3
                        params[2] = 'singular'
                        q.root['word'] = pattern.en.conjugate(saved, *params)
            if start == 'Where':
                start = 'What'
            first = [x for x in q.question if x['id'] not in rem][0]
            if start == 'When' and first['pos'] == 'VERB' and first['lemma'] not in ['be', 'do']:
                start = 'What'
            remaining = [x for x in q.question if x['id'] not in rem]
            imp = '%s %s?' % (start, ' '.join([x['word'] for x in remaining]))
            if self.is_answer_allowed(sub) and not self.filter_do(parsed_qa) and len(remaining) > 1:
                ret.append((imp, ' '.join(x['word'] for x in sub), 'subj'))
            q.root['word'] = saved
        if subj and dobj:
            dob = sorted([x for y in dobj for x in q.get_subtree(y)], key=lambda a:a['id'])
            # print(' '.join([x['word'] for x in sub]))
            # print(q.root['word'])
            # print(' '.join([x['word'] for x in dob]))
            dob_ids = [x['id'] for x in dob]
            spacy_span = parsed_qa.question_const_parse[min(dob_ids): max(dob_ids) + 1]
            # start = who_or_what(spacy_span)
            start = self.wh_picker(parsed_qa, dob_ids)
            p = pattern.en.tenses(q.root['word'])
            params = ['present', 3]
            if p:
                params = list(p[0])
            do = pattern.en.conjugate('do', *params)
            to_move = []
            if q.root['lemma'] == 'be':
                to_move.append(q.root)
            to_move.extend([x for x in q.get_children(q.root, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')])
            # if to_move:
            #     print(to_move)
            to_move = [x for x in to_move if x['lemma'] in ['be', 'can', 'do', 'will', 'would', 'have', 'may']]
            # if to_move:
            #     print(to_move)
            if any([x['dep'] == 'aux' for x in to_move]):
                to_move = [x for x in to_move if x['dep'] not in ['auxpass', 'advmod']]
            if any([x['lemma'] in ['be'] for x in to_move]):
                to_move = sorted([x for x in to_move if x['lemma'] in ['be']], key=lambda z:z['id'])[:1]
            if to_move:
                do = ' '.join([x['word'] for x in to_move])
                to_remove = to_remove.union(set([x['id'] for x in to_move]))
            else:
                q.root['word'] = pattern.en.conjugate(q.root['word'], 'infinitive')
            to_remove = to_remove.union(set(dob_ids))
            imp = '%s %s %s?' % (start, do, ' '.join([x['word'] for x in q.question if x['id'] not in to_remove]))
            if self.is_answer_allowed(dob):
                ret.append((imp, ' '.join(x['word'] for x in dob), 'dobj'))
            # print([q.root['word'],
        return ret

class ImplicationsVQA:
    def __init__(self, qa_parser=None):
        if qa_parser is not None:
            self.parser = qa_parser
        else:
            self.parser = QAParser()
        self.nlp = self.parser.nlp
        this_dir, _ = os.path.split(__file__)
        lm_path = os.path.join(this_dir, 'question_4gram.arpa')
        self.lm = language_model.LanguageModel(lm_path)
        self.score_product = self.lm.score_product
        self.get_best = lambda x:self.score_product(x)[0][0]

    def parse_dataset(self, questions, answers, **kwargs):
        return self.parser.parse_qas(questions, answers, add_declarative=False, **kwargs)

    def implications(self, question, answer):
        parsed = self.parser.parse_qa(question, answer)
        return self.implications_from_parsed(parsed)

    def implications_from_parsed(self, parsed_qa):
        ret = []
        if parsed_qa.question.startswith('What'):
            ret.extend(self.what(parsed_qa))
        if parsed_qa.question.startswith('How many'):
            ret.extend(self.howmany(parsed_qa))
        if parsed_qa.question.startswith('Where'):
            ret.extend(self.where(parsed_qa))
        if parsed_qa.question.startswith('What color'):
            ret.extend(self.color_in_answer_must_be_in_picture(parsed_qa))
        else:
            if parsed_qa.answer_const_parse[0].pos_ == 'NOUN':
                ret.extend(self.noun_in_answer_must_be_in_picture(parsed_qa))
        qwords = [x.text for x in parsed_qa.question_const_parse]
        qpos = [x.pos_ for x in parsed_qa.question_const_parse]
        qtag = [x.tag_ for x in parsed_qa.question_const_parse]
        if 'or' in qwords:
            ret.extend(self.xory(parsed_qa))
        if parsed_qa.answer == 'yes':
            if 'JJ' in qtag:
                ret.extend(self.adj_mutex(parsed_qa))
            modifiers = find_seq(parsed_qa.question_const_parse, ['NN', 'NN'])
            modifiers += find_seq(parsed_qa.question_const_parse, ['NN', 'NNS'])
            modifiers += find_seq(parsed_qa.question_const_parse, ['ADJ', 'NN'])
            modifiers += find_seq(parsed_qa.question_const_parse, ['ADJ', 'NNS'])

            if modifiers:
                ret.extend(self.remove_modifier(parsed_qa, modifiers))
        return ret


    def howmany(self, parsed_qa, use_lm=False):
        # Span is a span of ^ VERB NP $ inside an SQ
        qtext = parsed_qa.question
        answer = parsed_qa.answer_const_parse
        if not answer.text.isdigit():
            return []
        q = QuestionRepr(parsed_qa)
        q2 = QuestionRepr(parsed_qa)
        qsingular = QuestionRepr(parsed_qa)
        thing = q.question[q.question[1]['head']]
        thing2 = q2.question[q2.question[1]['head']]
        singular = answer.text == '1'
        sing = pattern.en.singularize(thing2['word'])
        thing2['word'] = sing if sing != thing2['word'] else thing2['lemma']
        things = sorted([x['id'] for x in q.get_subtree(thing)])[2:]
        things_text = ' '.join([x['word'] for x in sorted(q.get_subtree(thing), key=lambda a:a['id'])[2:]])
        things_text2 = ' '.join([x['word'] for x in sorted(q2.get_subtree(thing2), key=lambda a:a['id'])[2:]])
        things_t = things_text2 if singular else things_text
        if thing['head'] == -1:
            verb = 'Is' if singular else 'Are'
            ret = [('%s there %s %s' % (verb, answer.text, things_t), 'yes', 'yeseqcount')]
            verb = 'Are'
            if answer.text == '0':
                ret.append(('%s there %s %s' % (verb, int(answer.text) + 1, things_text2), 'no', 'n+1'))
                ret.append(('%s there any %s' % (verb, things_text), 'no', 'ans=0 implies none'))
            else:
                ret.append(('%s there %s %s' % (verb, int(answer.text) + 1, things_text), 'no', 'n+1'))
                ret.append(('%s there any %s' % (verb, things_text), 'yes', 'ans>0 implies some'))
            return ret
        else:
            root = q.question[thing['head']]
            root2 = q2.question[thing2['head']]
            to_move = []
            to_move2 = []
            if root['lemma'] == 'be':
                to_move.append(root)
                to_move2.append(root2)
            to_move.extend([x for x in q.get_children(root, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')])
            to_move2.extend([x for x in q2.get_children(root2, ['advmod', 'aux', 'auxpass', 'cop'], 'anywhere')])
            if any([x['dep'] == 'aux' for x in to_move]):
                to_move = [x for x in to_move if x['dep'] not in ['auxpass', 'advmod']]
                to_move2 = [x for x in to_move2 if x['dep'] not in ['auxpass', 'advmod']]
            be = [x for x in q2.question if x['lemma'] == 'be']
            for b in be:
                tense = 'past' if self.nlp.vocab.morphology.tag_map[b['tag']].get('Tense') == 'past' else 'present'
                b['word'] = pattern.en.conjugate(b['word'], tense=tense, person=3)
    #         if root['lemma'] == 'be':
    #             advmod = q.get_children(root, 'advmod', 'anywhere')
    #             root = [root] + advmod
    #         else:
    #             root = [root]
            to_remove = set([x['id'] for x in to_move] + [0, 1] + things + [x['id'] for x in q.question if x['word'] == '?'])
            bef = ' '.join([x['word'] for x in sorted(to_move, key=lambda a: a['id'])])
            bef2 = ' '.join([x['word'] for x in sorted(to_move2, key=lambda a: a['id'])])
            after = ' '.join([x['word'] for x in q.question if x['id'] not in to_remove])
            after2 = ' '.join([x['word'] for x in q2.question if x['id'] not in to_remove])
            if root['lemma'] == 'have':
                if not bef.lower().startswith('do'):
                    bef = 'Do '+  bef if bef else 'Do'
                    bef2 = 'Does '+  bef if bef else 'Does'

            bef_to_use = bef2 if singular else bef
            after_to_use = after2 if singular else after
            if thing['dep'] in ['dobj', 'dep', 'advcl']:
                before_p = '%s %s' % (bef, after)
                after_p = '%s' % things_text
                before_s = '%s %s' % (bef2, after2)
                after_s = '%s' % things_text2
                before = '%s %s' % (bef_to_use, after_to_use)
                after = '%s' % things_t
            else:
                before_p = '%s' % bef
                after_p = '%s %s' % (things_text, after)
                before_s = '%s' % bef2
                after_s = '%s %s' % (things_text2, after2)
                before = '%s' % bef_to_use
                after = '%s %s' % (things_t, after_to_use)

            before_s = capitalize(before_s)
            before_p = capitalize(before_p)
            before = capitalize(before)
            ret = [('%s %s %s?' % (before, answer.text, after), 'yes', 'yeseqcount')]
            if int(answer.text) == 0:
                ret.append(('%s %s %s?' % (before_s, int(answer.text) + 1, after_s), 'no', 'n+1'))
                ret.append(('%s any %s?' % (before_p, after_p), 'no', 'ans=0 implies none'))
            else:
                ret.append(('%s %s %s?' % (before_p, int(answer.text) + 1, after_p), 'no', 'n+1'))
                ret.append(('%s any %s?' % (before_p, after_p), 'yes', 'ans>0 implies some'))
            return ret

    def what(self, parsed_qa):
        # Span is a span of ^ VERB NP $ inside an SQ
        midfixes = ['', 'a', 'an', 'the']
        qtext = parsed_qa.question
        q = QuestionRepr(parsed_qa)
        answer = parsed_qa.answer_const_parse
        thing = q.question[0]
        if q.question[thing['head']]['pos'] != 'VERB' and thing['head'] != q.root['id']:
        # if thing['head'] != q.root['id']:
            thing = q.question[thing['head']]
            # print(thing)
        root = q.question[thing['head']]
        subjs = q.get_children(q.root, ['nsubj', 'csubj', 'nsubjpass'], 'anywhere')
        partmod = q.get_children(root, ['partmod'], 'right')
        answer_is_verbing = answer[0].tag_ == 'VBG' or answer[0].text.endswith('ing')
        # Verb, doing
        if partmod and partmod[0]['word'] == 'doing' and answer_is_verbing:
            to_remove = set([x['id'] for x in q.get_subtree(thing)])
            partmod[0]['word'] = answer.text
            bef = ' '.join([x['word'] for x in q.question if x['id'] not in to_remove])
            return [(bef, 'yes', 'what')]
        if ((thing['dep'] in ['nsubj', 'partmod', 'nsubjpass', 'csubj'] and q.root['pos'] != 'VERB') or
            thing['dep'] == 'dobj' or
            (thing['dep'] in ['nsubj', 'partmod', 'nsubjpass', 'csubj'] and len(subjs) > 1)) :
            to_remove = set([x['id'] for x in q.get_subtree(thing)] + [x['id'] for x in q.question if x['word'] == '?'])
            if q.root['lemma'] == 'do' and (answer_is_verbing):
                to_remove.add(q.root['id'])
            bef = ' '.join([x['word'] for x in q.question if x['id'] not in to_remove])
            new = self.score_product([bef, midfixes, answer.text, '?'])
            score = new[1][0]
            new = new[0][0]
            # prev = '%s %s? Yes' % (bef, answer.text)
            # if new.lower() != prev.lower():
            #     return 'BLA ' +new
            ret = [(new, 'yes', 'what')]
            if 'color' in qtext:
                opp = color_opposite(answer.text)
                new = self.get_best([bef, midfixes, opp]) + '?'
                ret.append((new, 'no', 'color mutex'))
            rel = []
            if answer[0].pos_ == 'NOUN':
                rel = get_related(answer.text, plural=answer[0].tag_ == 'NNS')
            # if answer[0].pos_ == 'ADJ':
            #     rel = get_antonyms(answer.text, 'ADJ')
            if rel and 'color' not in qtext:
                new = self.score_product([bef, midfixes, [x[0] for x in rel], '?'])
                nscore = new[1][0]
                new = new[0][0]
                if nscore - score > -1:
                    ret.append((new, 'no', 'wordnet mutex'))
            return ret
        if (thing['dep'] in ['nsubj', 'nsubjpass', 'csubj'] and q.root['pos'] == 'VERB'):
            things = sorted([x['id'] for x in q.get_subtree(thing)])
            things_text = ' '.join([x['word'] for x in sorted(q.get_subtree(thing), key=lambda a:a['id'])[2:]])
            # root = q.root
            to_move = []
            if root['lemma'] == 'be':
                to_move.append(root)
            to_move.extend([x for x in q.get_children(root, ['advmod', 'aux', 'auxpass', 'cop', 'dep'], 'anywhere')])
            if any([x['dep'] == 'aux' for x in to_move]):
                to_move = [x for x in to_move if x['dep'] not in ['auxpass', 'advmod']]
            if answer[0].tag_ == 'NNS':
                for x in to_move:
                    if x['lemma'] == 'be':
                        x['word'] = change_number(x['word'], plural=True)
            to_remove = set([x['id'] for x in to_move] + things + [x['id'] for x in q.question if x['word'] == '?'])
            if root['lemma'] == 'do':
                to_remove.add(root['id'])
            add_does = root['tag'] in ['VBZ', 'VPB'] and root['lemma'] not in ['be', 'do']
            if add_does:
                t = pattern.en.conjugate(root['word'], tense='infinitive')
                root['word'] = t if t else root['word']
            # if root['lemma'] not in ['be', 'do'] and thing['dep'] == 'nsubj' and q.aux is None:
            #     b = root
            #     # tense = 'past' if nlp.vocab.morphology.tag_map[b['tag']].get('Tense') == 'past' else 'present'
            #     tense = 'present'

            bef = ' '.join([x['word'] for x in sorted(to_move, key=lambda a: a['id'])])
            after = ' '.join([x['word'] for x in q.question if x['id'] not in to_remove])

            # if root['lemma'] not in ['be', 'do'] and thing['dep'] == 'nsubj' and q.aux is None:
            if add_does:
                p = 'Does' if answer[0].tag_ != 'NNS' else 'Do'
                if not bef.lower().startswith('do'):
                    new = ('%s?' % (' '.join(filter(lambda x:x, [p, answer.text, bef, after]))))
                    return [(new, 'yes', 'what')]
            new = self.score_product([bef, midfixes, answer.text, after, '?'])
            score = new[1][0]
            new = new[0][0]
            ret =  [(new, 'yes', 'what')]
            rel = []
            if answer[0].pos_ == 'NOUN':
                rel = get_related(answer.text, plural=answer[0].tag_ == 'NNS')
            # if answer[0].pos_ == 'ADJ':
            #     rel = get_antonyms(answer.text, 'ADJ')
            if rel:
                new = self.score_product([bef, midfixes, [x[0] for x in rel], after, '?'])
                nscore = new[1][0]
                new = new[0][0]
                if nscore - score > -1:
                    ret.append((new, 'no', 'wordnet mutex'))
            return ret
            # prev = '%s %s %s? Yes' % (bef, answer.text, after)
            # if new.lower() != prev.lower():
            #     return 'BLA ' +new.lower() + '\nBLA ' + prev

        return []

    def xory(self, parsed_qa):
        q = parsed_qa.question_const_parse
        answer = parsed_qa.answer_const_parse
        s = [i for i, x in enumerate(q) if x.text == 'or'][-1]
        e = s + 1
        seq_after = [x.pos_ for x in q[e:]]
        seq_before = [x.pos_ for x in q[:s]]
        starts = list(reversed([i for i in range(len(seq_after)) if seq_after[i] == seq_before[-1]]))
        match = None
        for x in starts:
            # print(x, seq_after[:x + 1], seq_before[(-x-1):])
            if seq_after[:x + 1] == seq_before[-x-1:]:
                match = x + 1
                break
        if len(seq_after) == 2:
            match = 1
        if match is None:
            return ''
        before = [x.text for x in q[:e-match - 1]]
        after = [x.text for x in q[e + match:]]
        mid1 = q[e-match - 1:e - 1].text
        mid2 = q[e:e + match].text
        in_1 = answer.text in mid1
        in_2 = answer.text in mid2
        if (in_1 and in_2) or not (in_1 or in_2):
            return ''
        mapz = {True: 'yes', False: 'no'}
        mapname = {True: 'xory_yes', False:'xory_no'}
        return [(' '.join(before + [mid1] + after), mapz[in_1], mapname[in_1]),
                (' '.join(before + [mid2] + after), mapz[in_2], mapname[in_2])]

    def adj_mutex(self, parsed_qa):
        q = parsed_qa.question_const_parse
        q2 = QuestionRepr(parsed_qa)
        jjs = [i for i, x in enumerate(q) if x.tag_ == 'JJ']
        banned = set(['other', 'same', 'green', 'clear', 'real'])
        ret = []
        for jj in jjs:
            thing = q2.question[jj]
            adj = q[jj].text
            if thing['dep'] != 'root' and q2.question[thing['head']]['dep'] in ['nsubj', 'prep', 'csubj', 'nsubjpass']:
                continue
            if adj in banned:
                continue
            a = get_antonyms(adj, 'ADJ')
            if a:
                a = [x[0] for x in a]
                before = q[:jj].text
                b = re.sub(r' an?$','', before)
                if b != before:
                    a = ['an ' + x if x[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+ x for x in a]
                    before = b
                tok = self.get_best([before, a, q[jj + 1:].text])
                ret.append((tok, 'no', 'wordnet_adj_mutex'))
        return ret

    def remove_modifier(self, parsed_qa, positions):
        q = parsed_qa.question_const_parse
        ret = []
        for p in positions:
            before = q[:p].text
            after = q[p + 1:].text
            banned = set(['other', 'same'])
            adj = q[p].text
            if adj in banned:
                return []
            b = re.sub(r' an?$','', before)
            if b != before:
                after = 'an ' + after if after[0] in ['a', 'e', 'i', 'o', 'u'] else 'a '+ after
                before = b
            ret.append((before + ' '+  after, 'yes', 'remove_modifier'))
        return ret

    def noun_in_answer_must_be_in_picture(self, parsed_qa):
        question = parsed_qa.question_const_parse
        answer = parsed_qa.answer_const_parse
        if 'country' in question.text or 'How many' in question.text or 'What type' in question.text:
            return []
        pref = 'Is' if answer[0].tag_ == 'NN' else 'Are'
        prefixes = ['%s there' % pref, 'Is anyone']#, 'Is it']
        midfixes = ['any', '', 'a', 'an']
        postfix = 'in the picture ?'
        if answer[0].tag_ not in ['NN', 'NNS']:
            return []
        q = self.score_product([prefixes, midfixes, answer.text, postfix])[0][0]
        return [(q, 'yes', 'noun_in_answer_must_be_in_picture')]

    def color_in_answer_must_be_in_picture(self, parsed_qa):
        question = parsed_qa.question_const_parse
        answer = parsed_qa.answer_const_parse
        q = 'Is there anything %s in the picture?' % answer.text
        return [(q, 'yes', 'color_in_answer_must_be_in_picture')]

    def where(self, parsed_qa):
        q = parsed_qa.question_const_parse
        answer = parsed_qa.answer_const_parse
        preps = ['on', 'in', 'at', 'on the', 'in the', 'at the', 'on a', 'in a', 'at a', 'on an', 'in an', 'at an', 'the', 'a', 'an']
        sp = q[1:].text.strip('?')
        if sp.startswith('\'s'):
            sp = 'Is' + sp[2:]
        if len(answer) != 1 or answer[0].pos_ in ['VERB', 'ADP', 'ADV']:
            preps += ['']
        q = self.score_product([sp, preps, insert_articles_in_answer(answer.text), '?'])[0][0]
        # print(sp, insert_articles_in_answer(answer.text))
        # print(q)
        # if q[1].text = '\'s'
        return [(q, 'yes', 'where')]
