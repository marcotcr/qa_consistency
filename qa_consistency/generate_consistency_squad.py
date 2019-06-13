import argparse
import json
import pickle
import pickle
import spacy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--implication_file', '-i', required=True)
    parser.add_argument('--orig_preds_path', '-p', required=True)
    parser.add_argument('--orig_dataset_path', '-z', required=True)
    parser.add_argument('--output_file', '-o', required=True)
    args = parser.parse_args()

    data, answers = load_datasets.load_squad('validation')
    answer_texts = [[x[0] for x in y] for y in answers]
    implications = pickle.load(open(args.imp_question_path, 'rb'))
    orig_preds = [json.loads(x)['best_span_str'] for x in open(args.orig_preds_path).readlines()]
    exact = np.array([b in a for b, a in zip(orig_preds, answer_texts)])
    new_data = []
    k = 0
    is_checked = set()
    edges = {}
    for i in exact.nonzero()[0]:
        answer = orig_preds[i]
        passage = data[i]['passage']
        if i in implications and answer in implications[i]:
            for q, new_a, t in implications[i][answer]:
    #             if i not in edges:
    #                 edges[i] = []
    #             edges[i].append(k)
                is_checked.add(i)
                k += 1
                new_data.append({'passage': passage, 'question': q, 'new_a': new_a, 'type': t, 'orig_question': data[i]['question'], 'orig_answer': answer})
    print('Exact match: %d (%.1f%% of preds)' % (exact.sum(), exact.mean() * 100))
    print('Checked for consistency: %d (%.1f%% of preds, %.1f%% of exact match)' % (len(is_checked), 100 * len(is_checked) / len(exact), 100* len(is_checked) / sum(exact)))
    open(args.output_file, 'w').write('\n'.join([json.dumps(x) for x in new_data]))

if __name__ == "__main__":
    main()
