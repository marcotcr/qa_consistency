#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import json
import os
import argparse
import collections
import re
import numpy as np
from . import implication

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def image_id_to_filename(image_id):
    strid = str(image_id)
    prefix = 'COCO_val2014_'
    return prefix + '0' * (12 - len(strid)) + strid + '.jpg'

def map_and_highest_id(dataset):
    questionmapper = {}
    highest_id = 0
    for q, a in dataset.question_attributes.items():
        tuple_ = (a['question'], a['image_id'])
        questionmapper[tuple_] = q
        highest_id = max(highest_id, q)
    return questionmapper, highest_id

def generate_implication_vqa(original_dataset, original_preds_path, implication_dict, output_path):
    source_map = {
     'ans=0 implies none' : 'logeq',
     'ans>0 implies some': 'necessary_condition',
     'color mutex': 'mutex',
     'color_in_answer_must_be_in_picture': 'necessary_condition',
     'n+1': 'mutex',
     'noun_in_answer_must_be_in_picture': 'necessary_condition',
     'remove_modifier': 'necessary_condition',
     'subjectyes': 'logeq',
     'what': 'logeq',
     'where': 'logeq',
     'whereprep': 'logeq',
     'wordnet mutex': 'mutex',
     'wordnet_adj_mutex': 'mutex',
     'xory_no': 'mutex',
     'xory_yes': 'logeq',
     'yeseqcount': 'logeq'
    }
    original = original_dataset
    original_preds = json.load(open(original_preds_path))
    original_preds = dict([(x['question_id'], x['answer']) for x in original_preds])
    new_dataset = Bunch({})
    new_dataset.question_mapper, new_dataset.highest_id = map_and_highest_id(original_dataset)
    new_dataset.annotations = {}
    new_dataset.questions = {}
    for idx, attributes in original.question_attributes.items():
        question = attributes['question']
        answer = original_preds[idx]
        implications = implication_dict.get((question, answer), [])
        image_id = attributes['image_id']
        for imp in implications:
            q, a, source = imp
            tuple_ = (q, image_id)
            if tuple_ not in new_dataset.question_mapper:
                new_idx = new_dataset.highest_id + 1
                new_dataset.question_mapper[tuple_] = new_idx
                new_dataset.highest_id += 1
            new_idx = new_dataset.question_mapper[tuple_]
            new_dataset.annotations[new_idx] = {
                'answer_type': 'yes/no',
                'answers': [{'answer': a, 'answer_confidence': 'yes', 'answer_id': i} for i in range(1, 11)],
                'image_id': image_id,
                'multiple_choice_answer': a,
                'question_id': new_idx,
                'question_type': source_map[source]
            }
            new_dataset.questions[new_idx] = {
                'question': question,
                'image_id': image_id,
                'question_id': new_idx
            }
    print('Writing:\n%s\n%s' % (os.path.join(output_path, 'questions.json'),
                                 os.path.join(output_path, 'annotations.json')))
    write_files(output_path, new_dataset)
def write_files(path, dataset):
    questions = list(dataset.questions.values())
    with open(os.path.join(path, 'questions.json'), 'w') as question_file:
        json.dump({'questions': questions}, question_file)
    annotations = list(dataset.annotations.values())
    with open(os.path.join(path, 'annotations.json'), 'w') as question_file:
        json.dump({'annotations': annotations}, question_file)

def evaluate_consistency_vqa(consistency_folder, consistency_preds_json):
    annotations = dict([(x['question_id'], x) for x in json.load(open(os.path.join(consistency_folder, 'annotations.json')))['annotations']])
    preds = json.load(open(consistency_preds_json))
    preds = dict([(x['question_id'], x['answer']) for x in preds])
    stats = collections.defaultdict(lambda: [])
    for i in preds:
        consistent = preds[i] == annotations[i]['multiple_choice_answer']
        source = annotations[i]['question_type']
        stats['all'].append(consistent)
        stats[source].append(consistent)
    for x in stats:
        stats[x] = np.mean(stats[x])
    return stats

def question_answers_product(questions, answer_lists):
    ret = set()
    for q, answers in zip(questions, answer_lists):
        for a in answers:
            ret.add((q, a))
    ret = list(ret)
    return [x[0] for x in ret], [x[1] for x in ret]

def question_answers_context_product(questions, answer_lists, context_list):
    ret = set()
    for q, answers, c in zip(questions, answer_lists, context_list):
        for a in answers:
            ret.add((q, a, c))
    ret = list(ret)
    return [x[0] for x in ret], [x[1] for x in ret], [x[2] for x in ret]

def load_vqa(path, fold='validation'):
    if fold == 'validation':
        j = json.load(open(path + '/OpenEnded_mscoco_val2014_questions.json'))
        j2 = json.load(open(path + '/mscoco_val2014_annotations.json'))
    elif fold == 'train':
        j = json.load(open(path + '/OpenEnded_mscoco_train2014_questions.json'))
        j2 = json.load(open(path + '/mscoco_train2014_annotations.json'))
    else:
        print('ERROR, fold must be validation or train')
        quit()
    question_map = {}
    question_attributes = {}
    for x in j['questions']:
        question_map[x['question_id']] = x['question'], image_id_to_filename(x['image_id']), x['image_id']
        question_attributes[x['question_id']] = x
    for x in j2['annotations']:
        q = question_map[x['question_id']]
        question_attributes[x['question_id']].update(x)
        question_map[x['question_id']] = (q[0], q[1], x['multiple_choice_answer'], x['answers'], q[2])
    validation = []
    for i, (question, filename, answer, ans, image_id) in question_map.items():
        validation.append((i, question, filename, answer, ans, image_id))
    validation = sorted(validation, key = lambda x:x[2])
    dataset = Bunch({})
    dataset.idxs = [x[0] for x in validation]
    dataset.questions = [x[1] for x in validation]
    dataset.filenames = [x[2] for x in validation]
    # dataset.paths = [os.path.join('/data/marcotcr/datasets/vqa/val2014', x) for x in dataset.filenames]
    dataset.answers = [x[3] for x in validation]
    dataset.all_answers = [[y['answer'] for y in x[4]] for x in validation]
    dataset.image_ids = [x[5] for x in validation]
    dataset.question_attributes = question_attributes
    return dataset

def load_vqav2(path, fold='validation'):
    if fold == 'validation':
        j = json.load(open(path + '/v2_OpenEnded_mscoco_val2014_questions.json'))
        j2 = json.load(open(path + '/v2_mscoco_val2014_annotations.json'))
        j3 = json.load(open(path + '/v2_mscoco_val2014_complementary_pairs.json'))
    elif fold == 'train':
        j = json.load(open(path + '/v2_OpenEnded_mscoco_train2014_questions.json'))
        j2 = json.load(open(path + '/v2_mscoco_train2014_annotations.json'))
        j3 = json.load(open(path + '/v2_mscoco_train2014_complementary_pairs.json'))
    else:
        print('ERROR, fold must be validation or train')
        quit()
    question_map = {}
    question_pair = {}
    question_attributes = {}
    for x in j['questions']:
        question_map[x['question_id']] = x['question'], image_id_to_filename(x['image_id']), x['image_id']
        question_attributes[x['question_id']] = x
    for x in j2['annotations']:
        q = question_map[x['question_id']]
        question_map[x['question_id']] = (q[0], q[1], x['multiple_choice_answer'], x['answers'], q[2])
        question_attributes[x['question_id']].update(x)
    for x in j3:
        question_pair[x[0]] = x[1]
        question_pair[x[1]] = x[0]
    validation = []
    for i, (question, filename, answer, ans, image_id) in question_map.items():
        pair = question_pair.get(i, None)
        validation.append((i, question, filename, answer, ans, pair, image_id))
    validation = sorted(validation, key = lambda x:x[2])
    dataset = Bunch({})
    dataset.idxs = [x[0] for x in validation]
    dataset.questions = [x[1] for x in validation]
    dataset.filenames = [x[2] for x in validation]
    # dataset.paths = [os.path.join('/data/marcotcr/datasets/vqa/val2014', x) for x in dataset.filenames]
    dataset.answers = [x[3] for x in validation]
    dataset.all_answers = [set([y['answer'] for y in x[4]]) for x in validation]
    idx_to_id = dict([(x, i) for i, x in enumerate(dataset.idxs)])
    dataset.pairs = [idx_to_id.get(x[5], None) for x in validation]
    dataset.image_ids = [x[6] for x in validation]
    dataset.question_attributes = question_attributes
    return dataset

def load_squad(path, fold='validation'):
    answers = []
    data = []
    ids = []
    files = {
        'validation': '/home/marcotcr/datasets/squad/dev-v1.1.json',
        'train': '/home/marcotcr/datasets/squad/train-v1.1.json',
        }
    f = json.load(open(files[fold]))
    for t in f['data']:
        for p in t['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                data.append({'passage': context, 'question': qa['question'], 'id': qa['id']})
                answers.append(set([(x['text'], x['answer_start']) for x in qa['answers']]))
    return data, answers

def generate_implication_squad(original_squad_path, original_preds_path, implication_dict, output_file):
    # original_squad_path: path for SQuAD json file, dev or training set
    # original_preds_path: path for model predictions in official SQuAD format (dictionary from id to prediction text)
    # implication_dict: dictionary from (question, answer, context) to implications
    # output_file: will write a json file in this path, using the same format as SQuAD
    dataset = json.load(open(original_squad_path))
    preds = json.load(open(original_preds_path))
    source_map = {}
    for t in dataset['data']:
        for p in t['paragraphs']:
            new_qas = []
            context = p['context']
            for qa in p['qas']:
                pred = preds[qa['id']]
                implications = implication_dict.get((qa['question'], pred, context), [])
                for i, imp in enumerate(implications):
                    q, a, source = imp
                    new_id = qa['id'] + '-%d' % i
                    source_map[new_id] = source
                    new_qas.append({'source': source, 'question': q, 'id': new_id, 'answers':[{'text': a}]})
            p['qas'] = new_qas
    json.dump(dataset, open(output_file, 'w'))
    # return dataset

def evaluate_consistency_squad(consistency_json, consistency_preds_json):
    dataset = json.load(open(consistency_json))
    preds = json.load(open(consistency_preds_json))
    process_answer = lambda x: implication.clean_answer(x.lower())
    stats = collections.defaultdict(lambda: [])
    source_map = lambda x: x if x not in set(['when', 'who', 'what', 'where', 'why']) else 'prep'
    for t in dataset['data']:
        for p in t['paragraphs']:
            for qa in p['qas']:
                pred = process_answer(preds[qa['id']])
                answer = process_answer(qa['answers'][0]['text'])
                consistent = pred in answer or answer in pred
                source = source_map(qa['source'])
                stats['all'].append(consistent)
                stats[source].append(consistent)
    for x in stats:
        stats[x] = np.mean(stats[x])
    return stats
    # return dataset

def squad_to_allennlp(squad_path, output_file):
    dataset = json.load(open(squad_path))
    out = []
    for t in dataset['data']:
        for p in t['paragraphs']:
            new_qas = []
            context = p['context']
            for qa in p['qas']:
                out.append(json.dumps({'passage': context, 'question' : qa['question'], 'id': qa['id']}))
    open(output_file, 'w').write('\n'.join(out))

def allennlp_preds_to_squad_format(input_file, pred_file, output_file=None):
    ids = [json.loads(x)['id'] for x in open(input_file)]
    preds = [json.loads(x)['best_span_str'] for x in open(pred_file)]
    ret = dict(list(zip(ids, preds)))
    if output_file is not None:
        json.dump(ret, open(output_file, 'w'))
    else:
        return ret
