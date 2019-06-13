#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

from setuptools import setup, find_packages

setup(name='qa_consistency',
      version='0.1',
      description='Generate implications to check the consistency of QA models',
      url='http://github.com/marcotcr/qa_consistency',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='MIT',
      install_requires=[
        'cython',
        'numpy',
        'benepar[gpu]',
        'spacy',
        'allennlp',
        'jupyter',
        'kenlm',
        'pattern',
        'mosestokenizer',
        'conllu',
      ],
      include_package_data=True,
      zip_safe=False)
