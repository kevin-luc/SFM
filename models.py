"""Implementation of an arbitrary order Factorization Machines."""
from __future__ import (absolute_import, division,
                                print_function, unicode_literals)
#from builtins import *
import os
import numpy as np
import tensorflow as tf
import shutil
from tqdm import tqdm


from .core import SFMCore
from .base import SFMBaseModel, loss_logistic, loss_mse




class SFMClassifier(SFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    Only binary classification with 0/1 labels supported.

    See SFMBaseModel docs for details about parameters.
    """

    def __init__(self, co_rank=10, view_rank = 0, isFullOrder=True, view_list=[[1]], input_type='dense', output_range=None, 
                n_epochs=100, optimizer=tf.train.AdamOptimizer(learning_rate=0.1), reg_type='L2', reg=0.1,
                batch_size=-1, init_scaling=2.0, log_dir=None, verbose=0,
                session_config=None):
        init_params = {
            'co_rank': co_rank,
            'view_rank': view_rank,
            'isFullOrder': isFullOrder,
            'view_list': view_list,
            'input_type': input_type,
            'output_range': output_range,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'reg_type': reg_type,
            'reg': reg,
            'init_scaling': init_scaling,
            'optimizer': optimizer,
            'log_dir': log_dir,
            'loss_function': loss_logistic,
            'verbose': verbose
        }
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        # suppose input {0, 1}, but use instead {-1, 1} labels
        assert(set(y_) == set([0, 1]))
        return y_ * 2 - 1

    def predict(self, X, mode_matrices = None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        raw_output = self.decision_function(X, mode_matrices)
        predictions = (raw_output > 0).astype(int)
        return predictions

    def predict_proba(self, X, mode_matrices = None):
        """Probability estimates.

        The returned estimates for all 2 classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        probs : array-like, shape = [n_samples, 2]
            Returns the probability of the sample for each class in the model.
        """
        outputs = self.decision_function(X, mode_matrices)
        probs_positive = utils.sigmoid(outputs)
        probs_negative = 1 - probs_positive
        probs = np.concatenate((probs_negative, probs_positive), axis=1)
        return probs


class SFMRegressor(SFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with MSE
    loss and gradient-based optimization.

    See SFMBaseModel docs for details about parameters.
    """
    def __init__(self, co_rank=10, view_rank = 0, isFullOrder=True, view_list=[[1]], input_type='dense', output_range = None,
                n_epochs=100, optimizer=tf.train.AdamOptimizer(learning_rate=0.1), reg_type='L2', reg=0.1,
                batch_size=-1, init_scaling=2.0, log_dir=None, verbose=0,
                session_config=None):
        init_params = {
            'co_rank': co_rank,
            'view_rank': view_rank,
            'isFullOrder': isFullOrder,
            'view_list': view_list,
            'input_type': input_type,
            'output_range': output_range,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'reg_type': reg_type,
            'reg': reg,
            'init_scaling': init_scaling,
            'optimizer': optimizer,
            'log_dir': log_dir,
            'loss_function': loss_mse,
            'verbose': verbose
        }
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        return y_

    def predict(self, X, mode_matrices = None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        predictions = self.decision_function(X, mode_matrices)
        return predictions
