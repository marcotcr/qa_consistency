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
      packages= find_packages(exclude=['js', 'node_modules', 'tests']),
      install_requires=[
        'allennlp==0.8.3',
        'benepar==0.1.2',
        'editdistance==0.5.3',
        'nltk==3.4.1',
        'numpy==1.22.0',
        'numpydoc==0.9.1',
        'Pattern==3.6',
        'scikit-learn==0.21.1',
        'scipy==1.3.0',
        'spacy==2.1.4',
        'tensorflow-estimator==1.13.0',
        'tensorflow-gpu==1.13.1',
        'torch==1.1.0',
        'overrides==1.9',
        # 'cython',
        # 'numpy',
        # 'benepar[gpu]',
        # 'spacy',
        # 'allennlp==2.1.0',
        # 'allennlp-models==2.1.0',
        # 'pattern',
        'jupyter',
        'kenlm',
        'mosestokenizer',
        'conllu',


      ],
      include_package_data=True,
      zip_safe=False)
