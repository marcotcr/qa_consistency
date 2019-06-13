#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

def is_aux(tok):
    return (tok['word'].lower() in AUX or tok['tag'] == 'MD' or tok['word'] == 'there')

def is_verb(tok):
    return (tok['tag'].startswith('V') or tok['pos'] in ['VERB', 'AUX'] or tok['tag'] == 'MD')

class QuestionRepr:
    def __init__(self, parsed_qa, use_declarative=False):
        const_parse = parsed_qa.question_const_parse
        dep_parse = parsed_qa.question_dep_parse
        if use_declarative:
            const_parse = parsed_qa.dec_const_parse
            dep_parse = parsed_qa.dec_dep_parse
        self.question = []
        # try:
        #     assert len(dep_parse['words']) == len(parsed)
        # except:
        #     print('FAIL')
        #     print('%d %s' % (len(dep_parse['words']), ' '.join(dep_parse['words'])))
        #     print('%d %s' % (len(parsed), parsed))
        for i, (t, w, h, d, p) in enumerate(zip(const_parse, dep_parse['words'], dep_parse['predicted_heads'], dep_parse['predicted_dependencies'], dep_parse['pos'])):
            self.question.append({
                'word':  w,
                'head': h - 1,
                'dep': d,
                'pos': t.pos_,
                'tag': p,
                'id': i,
                'lemma': t.lemma_
            })
        #
        self.root = self.get_node('root')
        #
        # # Get the subject
        subj_nodes = self.get_children(self.root, ['nsubj', 'nsubjpass', 'csubj'], 'anywhere')
        if len(subj_nodes) > 0:
            self.subj = subj_nodes[0]
        else:
            self.subj = None
        #
        # # Get auxiliary
        aux_nodes = self.get_children(self.root, ['aux', 'auxpass'], 'anywhere')
        if len(aux_nodes) > 0:
            self.aux = aux_nodes[0]
        else:
            self.aux = None

        self.obj = self.get_children(self.root, ['dobj', 'xcomp', 'ccomp', 'acomp', 'advmod', 'pobj'], 'anywhere')
        self.obj = self.obj[0] if self.obj else None
        # # Get copula
        self.cop = self.get_node('cop')
        #
        if self.cop is not None and self.cop['lemma'] == 'be':
            aux_nodes = self.get_children(self.root, ['aux'], 'anywhere')
            if len(aux_nodes) > 0 and aux_nodes[0]['tag'] == 'MD':
                self.aux = aux_nodes[0]
                self.root = self.cop
                self.cop = None
        #
        if self.cop is not None and self.aux is None and is_verb(self.root):
            self.aux = self.cop
            self.cop = None

    def get_node(self, rel):
        for tok in self.question:
            if tok['dep'].startswith(rel):
                return tok
        return None

    def get_all(self, rel):
        ret = []
        for tok in self.question:
            if tok['dep'].startswith(rel):
                ret.append(tok)
        return ret

    def get_subtree(self, node):
        subtree = []
        visited = set()
        to_visit = [node]
        while to_visit:
            v = to_visit.pop()
            if v['id'] not in visited:
                subtree.append(v)
                children = self.get_children(v, [],  'anywhere')
                to_visit.extend(children)
            visited.add(v['id'])
        return subtree

    def get_children(self, node, rels, loc='right'):
        assert (loc in ['left', 'right', 'anywhere'])
        head_id = node['id']
        children = []
        for tok in self.question:
            if 'head' in tok and tok['head'] == head_id \
                    and (len(rels) == 0 or tok['dep'] in rels):
                if loc == 'left' and tok['id'] < head_id:
                    children.append(tok)
                elif loc == 'right' and tok['id'] > head_id:
                    children.append(tok)
                elif loc == 'anywhere':
                    children.append(tok)
        return children
