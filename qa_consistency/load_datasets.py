#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import json
import os
import argparse
class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def image_id_to_filename(image_id):
    strid = str(image_id)
    prefix = 'COCO_val2014_'
    return prefix + '0' * (12 - len(strid)) + strid + '.jpg'

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
    for x in j['questions']:
        question_map[x['question_id']] = x['question'], image_id_to_filename(x['image_id']), x['image_id']
    for x in j2['annotations']:
        q = question_map[x['question_id']]
        question_map[x['question_id']] = (q[0], q[1], x['multiple_choice_answer'], x['answers'], q[2])
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
    return dataset

def load_squad(path, fold='validation'):
    answers = []
    data = []
    ids = []
    files = {
        'validation': os.path.join(path, 'dev-v1.1.json'),
        'train': os.paht.join(path, 'train-v1.1.json')
        }
    f = json.load(open(files[fold]))
    for t in f['data']:
        for p in t['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                data.append({'passage': context, 'question': qa['question'], 'id': qa['id']})
                answers.append(set([(x['text'], x['answer_start']) for x in qa['answers']]))
    return data, answers
