import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from utils import co_occurence_matrix_encoded, pmi
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from gensim.models import FastText
# from utils.text_encoders import MyBertTokenizer
from statistics import median_high

import numpy as np 
from six.moves import cPickle as pickle

from sklearn.preprocessing import normalize
import dask.array as da

n_components = 300

# This file computes the PPMI and FastText representations locally learnt on each class and on the whole dataset
def get_co_oc_corpora(root, cfile=None, tokenizer = 'spacy', min_occurences = 3):
    if cfile == None:
        all_files = [f for f in listdir('../' + root+"/0") if isfile(join('../' + root+"/0", f))]
        cf_lines = []
        for cf_n in all_files:
            cf = open(join('../' + root+"/0", cf_n), 'r')
            current_lines = cf.read().split('\n')
            current_lines = [line for line in current_lines if line not in ['', ' ', '\n']]
            cf_lines.extend(current_lines)

    else:
        cf = open(join('../' + root, cfile), 'r') 
        cf_lines = cf.read().split('\n')
        cf_lines = [line for line in cf_lines if line not in ['', ' ', '\n']]

    if cfile == None:
        category = 'dataset'
    else:
        category = cfile.strip('.txt')
        
    if tokenizer == 'spacy':
        encoder = SpacyEncoder(cf_lines, min_occurrences=min_occurences, append_eos=False, language ='fr')
    # if tokenizer == 'bert':
    #     encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)
    voc = {encoder.vocab[i] : i for i in range(len(encoder.vocab))}
    encoded_corpus = [encoder.encode(ex) for ex in cf_lines]
    tokenized_corpus = [[token.text for token in sent] for sent in encoder.spacy.pipe(cf_lines)]
    
    """
    The following has two parts: creating PPMI and creating fasttext representations
    """
    # Co_occurence counts
    median_length = median_high([len(enc) for enc in encoded_corpus])

    print('Getting co-occurence counts...')
    M = co_occurence_matrix_encoded(encoded_corpus, voc, median_length, False)
    PPMI = pmi(M)
    #PPMI_norm = normalize(PPMI, axis = 1, norm='l2')

    print('Applying SVD...')
    PPMI_da = da.from_array(PPMI)
    reduced_PPMI, _, _ = da.linalg.svd_compressed(PPMI_da, k= n_components)

    reduced_PPMI = np.array(reduced_PPMI)
    PPMI_proj = np.matmul(PPMI, reduced_PPMI)

    print('Writing file...')

    if not os.path.exists('./' + str(root)):
        os.makedirs('./' + str(root))
    Path('./' + str(root) + '/PPMI_' + str(category)).touch()
    with open('./' + str(root) + '/PPMI_' + str(category), 'wb') as f:
        pickle.dump(voc, f)
        pickle.dump(PPMI_proj, f)

    print('Creating Fasttext model')
    model = FastText(vector_size=n_components, window=median_length, min_count=min_occurences, workers=8)
    model.build_vocab(corpus_iterable=tokenized_corpus)
    model.train(corpus_iterable=tokenized_corpus, total_examples=model.corpus_total_words, epochs=50)

    vocab = model.wv.key_to_index
    matrix = model.wv.vectors

    print('Writing file...')
    if not os.path.exists('./' + str(root)):
        os.makedirs('./' + str(root))
    Path('./' + str(root) + '/FastText_' + str(category)).touch()
    model.save('./' + str(root) + '/FastText_' + str(category))

        
phases = ['train']

# 20Newsgroups
ag_subsets = ['comp', 'misc', 'pol', 'rec', 'rel', 'sci']

for phase in phases:
    for subset in ag_subsets:
        print("20 Newsgroups :",subset)
        get_co_oc_corpora(f'20ng_od/{phase}/0', f'{subset}.txt', min_occurences = 3)
    print("20 Newsgroups : dataset")
    get_co_oc_corpora(f'20ng_od/{phase}', min_occurences = 3)

# AG News
ag_subsets = ['business', 'sci', 'sports', 'world']

for phase in phases:
    for subset in ag_subsets:
        print("AG News :",subset)
        get_co_oc_corpora(f'ag_od/{phase}/0', f'{subset}.txt',min_occurences = 3)
    print("RNCP : dataset")
    get_co_oc_corpora(f'ag_od/{phase}',min_occurences = 3)

# Rncp
rncp_subsets = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']

for phase in phases:
    for subset in rncp_subsets:
        print("RNCP :",subset)
        get_co_oc_corpora(f'rncp_od/{phase}/0', f'{subset}.txt',min_occurences = 3)
    print("RNCP : dataset")
    get_co_oc_corpora(f'rncp_od/{phase}',min_occurences = 3)
    