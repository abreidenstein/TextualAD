import json
import logging
import time
import torch
import numpy as np
import pickle as pkl

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics.pairwise import pairwise_distances
from base.base_dataset import BaseADDataset
from networks.main import build_network


class OCSVM(object):
    """A class for One-Class SVM models."""

    def __init__(self, kernel='linear', nu=0.1):
        """Init OCSVM instance."""

        self.kernel = kernel
        self.nu = nu
        self.rho = None
        self.gamma = None

        self.model = OneClassSVM(kernel=kernel, nu=nu)
        self.embedding = None

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None
        }
    # Modification: adding cont argument for contamination of training data (unsupervised learning)
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
        self.pretrained_word_vectors = pretrained_word_vectors

        if not pretrained_word_vectors.startswith("DATE") and not pretrained_word_vectors.startswith("Electra_pretrained"):
            self.embedding = self.embedding.to(device)

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Trains the OC-SVM model on the training data."""
        logger = logging.getLogger()

        train_loader, _ = dataset.loaders(batch_size=32, num_workers=n_jobs_dataloader)

        # Training
        logger.info('Starting training...')

        if not self.pretrained_word_vectors.startswith("DATE"):
            X = ()
            for data in train_loader:
                _, text, _, weights = data
                if len(text.shape) == 1:
                    text = torch.unsqueeze(text, 0)
                text, weights = text.to(device), weights.to(device)
                if len(weights.shape) == 1:
                    weights = torch.unsqueeze(weights, 0)
                if self.pretrained_word_vectors.startswith("Electra_pretrained"):
                    X_batch = self.embedding(text)
                else:    
                    X_batch = self.embedding(text, weights)
                X += (X_batch.cpu().data.numpy(),)
            X = np.concatenate(X)
        else :
            if self.embedding == "DATE_class":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/train/"+str(dataset.cont)+"/DATE_"+dataset.normal_class_name
            if self.embedding == "DATE_dataset":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/train/DATE_Dataset"
            if self.embedding == "DATE_dataset_noRMD":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/train/DATE_Dataset_noRMD"
            if self.embedding == "DATE_class_noRMD":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/train/"+str(dataset.cont)+"/DATE_"+dataset.normal_class_name+"_noRMD"
            with open(reprFile, 'rb') as f:
                    X = pkl.load(f)

        # if rbf-kernel, re-initialize svm with gamma minimizing the numerical error
        if self.kernel == 'rbf':
            self.gamma = 1 / (np.max(pairwise_distances(X)) ** 2)
            self.model = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)


        start_time = time.time()
        self.model.fit(X)
        self.results['train_time'] = time.time() - start_time

        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the OC-SVM model on the test data."""
        logger = logging.getLogger()
        loggerSimp = logging.getLogger('simp')

        _, test_loader = dataset.loaders(batch_size=32, num_workers=n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')

        idx_label_score = []
        X = ()
        idxs = []
        labels = []
        if not self.pretrained_word_vectors.startswith("DATE"):
            for data in test_loader:

                idx, text, label_batch, weights = data
                
                if len(text.shape) == 1:
                    text = torch.unsqueeze(text, 0)
                if len(weights.shape) == 1:
                        weights = torch.unsqueeze(weights, 0)
                text = text.to(device)
                weights = weights.to(device)
                if self.pretrained_word_vectors.startswith("Electra_pretrained"):
                    X_batch = self.embedding(text)
                else:    
                    X_batch = self.embedding(text, weights)  # X_batch.shape = (batch_size, embedding_size)
                X += (X_batch.cpu().data.numpy(),)
                label_batch = label_batch.to(device)
            
                idxs += idx
                labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()

            X = np.concatenate(X)
        else :
            if self.embedding=="DATE_class":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/test/DATE_"+dataset.normal_class_name
            if self.embedding=="DATE_class_noRMD":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/test/DATE_"+dataset.normal_class_name+"_noRMD"
            if self.embedding=="DATE_dataset":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/test/DATE_Dataset_"+dataset.normal_class_name
            if self.embedding=="DATE_dataset_noRMD":
                reprFile = "../data/word_vectors_saved/"+dataset.dataset_name_files+"/test/DATE_Dataset_"+dataset.normal_class_name+"_noRMD"
            with open(reprFile, 'rb') as f:
                    X = pkl.load(f)
                    labels = pkl.load(f)
        
        start_time = time.time()
        scores = (-1.0) * self.model.decision_function(X)
        self.results['test_time'] = time.time() - start_time

        scores = scores.flatten()
        self.rho = -self.model.intercept_[0]

        if not self.pretrained_word_vectors.startswith("DATE"):
            #Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(idxs, labels, scores.tolist()))
            self.results['test_scores'] = idx_label_score

        # Compute AUC
        if not self.pretrained_word_vectors.startswith("DATE"):
            _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)
        precision_in,recall_in, _ = precision_recall_curve(labels, -scores, pos_label=0)
        self.results['test auprIn'] = auc(recall_in,precision_in)

        precision_out, recall_out, pr_thresholds_anom = precision_recall_curve(labels, scores)
        self.results['test auprOut'] = auc(recall_out, precision_out)


        # Log results
        loggerSimp.info('Test AUC: {:.10f}'.format(100. * self.results['test_auc']))
        loggerSimp.info('Test AUPR-in: {:.10f}'.format(100. * self.results['test auprIn']))
        loggerSimp.info('Test AUPR-out: {:.10f}'.format(100. * self.results['test auprOut']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')
        loggerSimp.info('------------')

    def save_model(self, export_path):
        """Save OC-SVM model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load OC-SVM model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
