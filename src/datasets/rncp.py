from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk import word_tokenize
#from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights
from sklearn.feature_extraction.text import CountVectorizer
from transformers import ElectraTokenizerFast
import numpy as np

import torch

maxLenForElectra = 512

class Rncp_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False, cont=None,useBow = False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.dataset_name = 'rncp'
        self.dataset_name_files = 'rncp_od'
        self.cont = cont
        classes = list(range(16))

        rncp_subsets = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']

        self.normal_class = normal_class
        self.normal_class_name = rncp_subsets[normal_class]
        self.outlier_classes = []
        del classes[normal_class]
        for i in classes:
            self.outlier_classes += rncp_subsets[i]

        # Load the dataset
        self.train_set, self.test_set = rncp_dataset(directory=root, train=True, test=True, clean_txt=clean_txt, cont=cont)
        
        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            if row['label'] == rncp_subsets[self.normal_class]:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.test_set):
            row['label'] = torch.tensor(0) if row['label'] == rncp_subsets[self.normal_class] else torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)
        
        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos, language ='fr')
        # if tokenizer == 'bert':
        #     self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)
        if tokenizer == 'electra':
            self.encoder = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator',cache_dir='../data/electra_cache')

        if useBow: #useBow is used by LSA and TONMF to compute the Bag-of-words representations of the train and test set
            self.trainBow = bow(self.train_set,self.encoder.vocab)
            self.testBow = bow(self.test_set,self.encoder.vocab)
        self.testIdxs,self.testLabels = labels(self.test_set)

        # Encode
        for row in datasets_iterator(self.train_set, self.test_set):
            if tokenizer == "electra":
                if len(row['text']) > maxLenForElectra:
                    splittedRow = row['text'].split(" ")[:maxLenForElectra]
                    row["text"] = " ".join(splittedRow)
                row['text'] = torch.tensor(self.encoder.encode(row['text'], add_special_tokens=True,truncation = True, padding = 'max_length'))
            elif append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i

# computes the Bag-of-words representations of the train and test set
def bow(dataset, voc):
    corpus = []
    for row in datasets_iterator(dataset):
        corpus.append(row['text'])
    vectorizer = CountVectorizer(vocabulary= voc)
    X = vectorizer.fit_transform(corpus)
    return(X)

def labels(dataset):
    labels = []
    for row in datasets_iterator(dataset):
        labels.append(row['label'])
    return(np.arange(len(labels)),labels)

def rncp_dataset(directory='../data', train=False, test=False, clean_txt=False, cont=None):
    """
    Load the RNCP dataset.

    Args:
        directory (str, optional): Directory to get the dataset from
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        cont (int, optional): corresponds to the contamination rate of the dataset for uncupervised setting.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]
    directory += '/rncp_od/'

    for split_set in splits:

        rncp_subsets = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']
        if cont is None: cont = 0
        if (cont is not None) and (split_set == 'train'):
            rncp_subsets = [ str(cont) + '/' + c for c in rncp_subsets ]
        
        all_files = [directory + split_set + '/' + subset + '.txt' for subset in rncp_subsets]
        cf_lines = []
        for cf_n in all_files:
            cf = open(cf_n, 'r')
            current_lines = cf.read().split('\n')
            current_lines = [line for line in current_lines if line not in ['', ' ', '\n']]
            cf_lines.append(current_lines)

        examples = []

        rncp_subsets = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']
        for i, subset in enumerate(cf_lines):
            for id in range(len(subset)):
                if clean_txt:
                    text = clean_text(subset[id])
                else:
                    text = ' '.join(word_tokenize(subset[id]))
                label = rncp_subsets[i]

                if text:
                    examples.append({
                        'text': text,
                        'label': label
                    })
                    
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
