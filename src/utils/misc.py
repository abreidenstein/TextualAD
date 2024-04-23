from .visualization.attention_viz import createHTML
from torch.utils.data.dataset import Dataset
from torchnlp.encoders.encoder import Encoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode

import numpy as np
import string
import re

import matplotlib.pyplot as plt
from math import log
import numpy as np


def print_text_samples(dataset: Dataset, encoder: Encoder, indices, export_file, att_heads=None, weights=None,
                       title=''):
    """Print text samples of dataset specified by indices to export_file text file."""

    export_txt = export_file + '.txt'
    txt_file = open(export_txt, 'w')# w to erase file when writing to it

    if title:
        txt_file.write(f'{title}\n\n')

    texts = []
    texts_weights = []
    i = 1
    for idx in indices:
        tokens = dataset[idx]['text']
        text = encoder.decode(tokens)

        if att_heads is not None:
            att_head = att_heads[i-1]
            txt_file.write(f'{i:02}. (h{att_head:02})\n {text}\n\n')
        else:
            txt_file.write(f'{i:02}.\n {text}\n\n')

        if weights is not None:
            texts_weights.append(weights[i-1][:len(tokens)])
        texts.append(text)

        i += 1

    txt_file.close()

    if weights is not None:
        export_html = export_file + '.html'
        createHTML(texts, att_heads, texts_weights, export_html)

    return

def print_text_samples_with_scores(dataset: Dataset, encoder: Encoder, indices, scores,sort_idx,labels, export_file, 
                       title=''):
    """Print text samples of dataset specified by indices to export_file text file with their anomaly score."""

    export_txt = export_file + '.txt'
    txt_file = open(export_txt, 'w')

    if title:
        txt_file.write(f'{title}\n\n')

    texts = []
    i = 0
    for idx in indices:
        if idx < len(indices):
            tokens = dataset[idx]['text']
            text = encoder.decode(tokens)
            score = scores[sort_idx[i]]
            label = labels[sort_idx[i]]

        txt_file.write(f'{i:02}. Real label : {label} - Index : {idx} - score : {score} \n {text}\n\n')

        texts.append(text)


        i += 1
    histOfScores(scores, labels,'../log/test_newsgroups20/',"testHistFr")
    txt_file.close()

    return
directoryForResults = '../log/test_newsgroups20/'

def histOfScores(scores, labels, directoryForResults,title):
    """To get a histogram of the scores of a dataset separated between normal class elements and anomalies."""
    scoresPositive = []
    scoresNegative = []
    for i in range(len(labels)):
        if labels[i] ==1:
            scoresPositive.append(scores[i])
        else:
            scoresNegative.append(scores[i])
    plt.clf()
    plt.figure(figsize=(10,8), dpi=80)
    plt.hist(scoresPositive,alpha=0.5,bins=100)
    plt.hist(scoresNegative,alpha=0.5,bins=100, color="red")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Anomaly score",fontsize=22)
    plt.ylabel("Number of documents",fontsize=22)
    plt.tight_layout()
    plt.savefig(directoryForResults+title+'.png')

def print_top_words(top_words_list, export_file, title=''):
    """
    Print the top words per context.
    :param top_words_list: list of lists of pairs (<word>, <count>) of top-k words in vocabulary by occurrence counts.
    :param export_file: path to export file.
    :param title: optional title.
    """

    n_contexts = len(top_words_list)

    export_txt = export_file + '.txt'
    txt_file = open(export_txt, 'a')

    if title:
        txt_file.write(f'{title}\n\n')

    for context in range(n_contexts):
        txt_file.write(f'Context {context:02}\n')

        for (word, count) in top_words_list[context]:
            txt_file.write(f'#{count:03}: {word}\n')

        txt_file.write(f'\n')

    txt_file.close()

    return


def get_correlation_matrix(a, eps=1e-08):
    """
    Compute correlation matrix (cosine similarity) of 2D-array a.
    The diagonal holds the norms of the row vectors (instead of cosine similarities of 1).
    """

    dot_products = np.dot(a, a.transpose())
    norms = np.sqrt(dot_products).diagonal()
    corr_mat = dot_products / (norms[:, np.newaxis] + eps)
    corr_mat = corr_mat / (norms[np.newaxis, :] + eps)
    corr_mat[np.diag_indices(len(norms))] = norms

    return corr_mat


def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True, language = 'english'):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words(language))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)
        if language == 'french': # needed for rncp dataset
           text = unidecode.unidecode(text)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text
