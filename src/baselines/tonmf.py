import json
import logging
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import check_random_state
from base.base_dataset import BaseADDataset
from numpy.linalg import norm


# This code was adapted from Ramakrishnan Kannan, Hyenkyun Woo, Charu C. Aggarwal, and Haesun Park. 2017. Outlier Detection for Text Data.
# Their code was in Matlab, available here : https://github.com/ramkikannan/outliernmf


class TONMF(object):
    def __init__(self, n_components = 10, alpha=1, beta=1, n_iter=10, n_update_wh=10, random_state=42, tolerance=1e-5):        
        self.n_components = n_components
        self.alpha = alpha
        print("alpha : ",self.alpha)
        self.beta = beta
        self.n_iter = n_iter
        self.n_update_wh = n_update_wh
        
        self.tolerance = tolerance
        self.epsilon = 1e-10
        
        self.random_state = random_state

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None
        }
        print("intialisé !")
        
    def update_wh(self,D,updateW, WifNotUpdated = None):
        D = self.A
        current_error = norm(D - self.W @ self.H, 'fro')
        for iteration in range(self.n_update_wh):
            if updateW:
                # UPDATE W
                DHt = D @ self.H.T
                HHt = self.H @ self.H.T
                # to avoid divide by zero error
                HHtDiag = np.diag(HHt)
                HHtDiag.setflags(write=1)
                HHtDiag[HHtDiag == 0] = self.epsilon
                for j in range(self.n_components):
                    Wj = self.W[:, j] + (DHt[:, j] - self.W @ HHt[:, j]) / HHtDiag[j]
                    Wj[Wj < self.epsilon] = self.epsilon
                    self.W[:,j] = Wj
            else :
                self.W = WifNotUpdated
            
            # UPDATE H
            WtD = self.W.T @ D
            WtW = self.W.T @ self.W
            # to avoid divide by zero error
            WtWDiag = np.diag(WtW)
            WtWDiag.setflags(write=1)
            WtWDiag[WtWDiag == 0] = self.epsilon
            for j in range(self.n_components):
                Hj = self.H[j, :] + (WtD[j, :] - WtW[j, :] @ self.H) / WtWDiag[j]
                Hj = Hj - self.beta / WtWDiag[j]
                Hj[Hj < self.epsilon] = self.epsilon
                self.H[j, :] = Hj
            error = norm(D - self.W @ self.H, 'fro')
            delta_error = error - current_error
            if np.abs(delta_error) <= self.tolerance:
                break
            current_error = error
            
    def _fit(self, X, updateW, WifNotUpdated = None,y=None):
        print("alpha from fit : ", self.alpha)
        self.n_rows, self.n_cols = X.shape
        
        self.A = X
        rs = check_random_state(self.random_state)
        self.W = rs.rand(self.n_rows, self.n_components)
        rs = check_random_state(self.random_state)
        self.H = rs.rand(self.n_components, self.n_cols)
        D = self.A - self.W @ self.H
        
        error_values = []
        
        for iteration in range(self.n_iter):
            colnormdi = np.sqrt(sum(np.power(D, 2)))
            colnormdi_factor = colnormdi - self.alpha
            colnormdi_factor[colnormdi_factor < 0] = 0
            
            colnormdiWithCorrectFormat = np.tile(np.array(colnormdi), (D.shape[0], 1))
            self.Z = np.divide(D, colnormdiWithCorrectFormat)
            self.Z = np.multiply(self.Z, colnormdiWithCorrectFormat)
            
            D = self.A - self.Z
            self.update_wh(D,updateW, WifNotUpdated)
            D = self.A - self.W @ self.H
            
            error = norm(D - self.Z, 'fro') + self.alpha * np.sum(np.sqrt(np.sum(np.power(self.Z,2), axis=0))) + self.beta * norm(self.H,1)
            error_values.append(error)
            
        self.error_values = error_values
        
    def fit_transform(self, X, updateW, WifNotUpdated = None, y=None):
        self._fit(X, updateW, WifNotUpdated, y)
        return self.W, self.H, self.Z

    
    def train(self, dataset: BaseADDataset):
        """Trains the TONMF model on the training data."""
        logger = logging.getLogger()

        # Training
        start_time = time.time()
        logger.info('Starting training...')
        logger.info('Applying TONMF...')
        self.WTrain, self.HTrain, self.ZTrain = self.fit_transform(np.transpose(dataset.trainBow),updateW = True)
        self.results['train_time'] = time.time() - start_time

        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the TONMF model on the test data."""
        logger = logging.getLogger()
        loggerSimp = logging.getLogger('simp')

        # Testing
        start_time = time.time()
        logger.info('Starting testing...')

        self.WTest, self.HTest, self.ZTest = self.fit_transform(np.transpose(dataset.trainBow),updateW = False,WifNotUpdated = self.WTrain)
        scores = norm(self.ZTest,ord = 2, axis = 1)

        # In the article of Kannan et al, the more an example is an outlier, the more its embedding is a combination of many vectors and thus with a higher norm.
        # Therefore the norm is multiplied by -1 to get a membership score.
        scores = scores*(-1) 

        idx_label_score = []
        idx_label_score += list(zip(list(dataset.testIdxs), list(dataset.testLabels), scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)
        precision_in,recall_in, _ = precision_recall_curve(labels, scores)
        self.results['test auprIn'] = auc(recall_in,precision_in)

        precision_out, recall_out, pr_thresholds_anom = precision_recall_curve(labels, -scores, pos_label=0)
        self.results['test auprOut'] = auc(recall_out, precision_out)

        self.results['test_time'] = time.time() - start_time

        # Log results
        logger.info('Test AUC: {:.10f}%'.format(100. * self.results['test_auc']))
        loggerSimp.info('Test AUC: {:.10f}'.format(100. * self.results['test_auc']))
        logger.info('Test AUPR-in: {:.10f}%'.format(100. * self.results['test auprIn']))
        loggerSimp.info('Test AUPR-in: {:.10f}'.format(100. * self.results['test auprIn']))
        logger.info('Test AUPR-out: {:.10f}%'.format(100. * self.results['test auprOut']))
        loggerSimp.info('Test AUPR-out: {:.10f}'.format(100. * self.results['test auprOut']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')
        loggerSimp.info('------------')

    def save_model(self, export_path):
        """Save TONMF model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load TONMF model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp,default=str)
