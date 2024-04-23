import json
import logging
import time
import numpy as np

import dask.array as da

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from base.base_dataset import BaseADDataset
from networks.main import build_network
from numpy import linalg as LA 

class LSA(object):
    """A class for LSA models."""

    def __init__(self, n_components=300, seed = 0):
        """Init LSA instance."""

        self.n_components = n_components
        self.model = da.linalg.svd_compressed
        self.embedding = None
        self.seed = seed

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None
        }
    # Modification: adding cont argument
    def set_embedding(self, dataset, embedding_size=100, pretrained_word_vectors=None, embedding_reduction='mean',
                      use_tfidf_weights=False, normalize_embedding=False, device: str = 'cpu', cont=None):
        """Sets the word embedding for the text data."""
        self.embedding = build_network('embedding',
                                       dataset,
                                       embedding_size=embedding_size,
                                       pretrained_model=pretrained_word_vectors,
                                       update_embedding=False,
                                       embedding_reduction=embedding_reduction,
                                       use_tfidf_weights=use_tfidf_weights,
                                       normalize_embedding=normalize_embedding,
                                       cont=cont)
        self.embedding = self.embedding.to(device)

    def train(self, dataset: BaseADDataset):
        """Trains the LSA model on the training data."""
        logger = logging.getLogger()

        # Training
        start_time = time.time()
        logger.info('Starting training...')

        logger.info('Applying SVD...')

        #The bag of words is computed as the dataset is loaded by executing the code in src/datasets/...
        bowDa = da.from_array(dataset.trainBow) 
        U, S, V = self.model(bowDa, k= self.n_components,seed=self.seed)
        self.U, self.S, self.V = np.array(U), np.array(S), np.array(V)

        self.results['train_time'] = time.time() - start_time

        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset):
        """Tests the LSA model on the test data."""
        logger = logging.getLogger()
        loggerSimp = logging.getLogger('simp')

        # Testing
        start_time = time.time()
        logger.info('Starting testing...')

        vectForScore = dataset.testBow - dataset.testBow @ np.transpose(self.V) @ np.diagflat(self.S) @ self.V
        scores = LA.norm(vectForScore,ord = 2, axis = 1)

        # The higher the norm, the more vectors are combined in the representation of the vector.
        # Therefore the results are multiplied by -1 to get a membership score.
        scores = scores*(-1) 

        # Save triples of (idx, label, score) in a list
        idx_label_score = []
        idx_label_score += list(zip(list(dataset.testIdxs), list(dataset.testLabels), scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)
        precision_in,recall_in, _ = precision_recall_curve(labels, -scores, pos_label=0)
        self.results['test auprIn'] = auc(recall_in,precision_in)

        precision_out, recall_out, _ = precision_recall_curve(labels, scores)
        self.results['test auprOut'] = auc(recall_out, precision_out)

        self.results['test_time'] = time.time() - start_time

        # Log results
        loggerSimp.info('Test AUC: {:.10f}'.format(100. * self.results['test_auc']))
        loggerSimp.info('Test AUPR-in: {:.10f}'.format(100. * self.results['test auprIn']))
        loggerSimp.info('Test AUPR-out: {:.10f}'.format(100. * self.results['test auprOut']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')
        loggerSimp.info('------------')

    def save_model(self, export_path):
        """Save LSA model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load LSA model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp,default=str)
