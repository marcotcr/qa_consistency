#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import kenlm
import itertools
import numpy as np
class LanguageModel:
    def __init__(self, path=None):
        self.lm = kenlm.Model(path)
    def list_fill(self, tokens_before, tokens_after, tokens_mid):
        words = []
        scores = []
        for mid in tokens_mid:
            tokens = tokens_before + mid + tokens_after# + ['<eos>']
            score = self.lm.score(' '.join(tokens))
            scores.append(score)
            words.append(' '.join(tokens))

        scores = np.array(scores)
        sort_order = np.argsort(-scores)
        return [words[i] for i in sort_order], [scores[i] for i in sort_order]
    def score_product(self, list_of_lists, capitalize=True):
        remove_empty = lambda x: filter(lambda y: y != '', x)
        list_of_lists = [[x] if type(x) != list else x for x in list_of_lists]
        all_options = itertools.product(*list_of_lists)
        all_options = [' '.join(remove_empty(x)) for x in all_options]
        if capitalize:
            all_options = [x[0].capitalize() + x[1:] for x in all_options]
        scores = np.array([self.lm.score(x) for x in all_options])
        sort_order = np.argsort(-scores)
        return [all_options[i] for i in sort_order], [scores[i] for i in sort_order]
