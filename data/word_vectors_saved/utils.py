from nltk import word_tokenize
import numpy as np 
import re

def vocabulary(corpus, count_threshold=1, voc_threshold=0):
    word_counts = {}
    for sent in corpus:
        sent = sent.lower()
        sent = re.sub(r"[^a-zA-Z0-9]+", " ", sent)
        for word in word_tokenize(sent):
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1    
    filtered_word_counts = {word: count for word, count in word_counts.items() if count >= count_threshold}        
    words = sorted(filtered_word_counts.keys(), key=word_counts.get, reverse=True)
    if voc_threshold > 0:
        words = words[:voc_threshold]
    words = words + ['UNK']   
    vocabulary = {words[i] : i for i in range(len(words))}
    return vocabulary, {word: filtered_word_counts.get(word, 0) for word in vocabulary}

def co_occurence_matrix(corpus, vocabulary, window=0, distance_weighting=False):
    l = len(vocabulary)
    M = np.zeros((l,l))
    for sent in corpus:
        sent = sent.lower()
        sent = re.sub(r"[^a-zA-Z0-9]+", " ", sent)
        sent = word_tokenize(sent)
        sent_idx = [vocabulary.get(word, len(vocabulary)-1) for word in sent]
        for i, idx in enumerate(sent_idx):
            if idx > -1:
                if window > 0:
                    l_ctx_idx = [sent_idx[j] for j in range(max(0,i-window),i)]                
                else:
                    l_ctx_idx = sent_idx[:i]                
                for j, ctx_idx in enumerate(l_ctx_idx):
                    if ctx_idx > -1:
                        if distance_weighting:
                            weight = 1.0 / (len(l_ctx_idx) - j)
                        else:
                            weight = 1.0
                        M[idx, ctx_idx] += weight * 1.0
                        M[ctx_idx, idx] += weight * 1.0
    return M  

def co_occurence_matrix_encoded(corpus, vocabulary, window=0, distance_weighting=False):
    l = len(vocabulary)
    M = np.zeros((l,l))
    for k, sent in enumerate(corpus):
        for i, idx in enumerate(sent):
            if idx > -1:
                if window > 0:
                    l_ctx_idx = [sent[j] for j in range(max(0,i-window),i)]
                else:
                    l_ctx_idx = sent[:i]
                for j, ctx_idx in enumerate(l_ctx_idx):
                    if ctx_idx > -1:
                        if distance_weighting:
                            weight = 1.0 / (len(l_ctx_idx) - j)
                        else:
                            weight = 1.0
                        M[idx, ctx_idx] += weight * 1.0
                        M[ctx_idx, idx] += weight * 1.0
    return M

def pmi(co_oc, positive=True):
    sum_vec = co_oc.sum(axis=0) + 1.0
    sum_tot = sum_vec.sum()
    with np.errstate(divide='ignore'):
        pmi = np.log((co_oc * sum_tot) / (np.outer(sum_vec, sum_vec)))                   
    pmi[np.isinf(pmi)] = 0.0  # log(0) = 0
    if positive:
        pmi[pmi < 0] = 0.0
    return pmi
