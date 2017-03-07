from __future__ import (absolute_import, division,
                                print_function, unicode_literals)
import tensorflow as tf
from .core import SFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Predefined loss functions
# Should take 2 tf.Ops: outputs and targets and should return tf.Op of loss
# Be carefull about dimentionality -- maybe tf.transpose(outputs) is needed

def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)), name='logit_loss')
    return raw_loss
#    return tf.minimum(raw_loss, 100, name='truncated_log_loss')

def loss_mse(outputs, y):
    return tf.pow(y -  tf.transpose(outputs), 2, name='mse_loss')
#    return tf.pow(y -  outputs, 2, name='mse_loss')


def batcher(X_, y_=None,  batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    perm: the permuation of the instances

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input
    ret_y : np.array or None, shape (batch_size,)
    """
    assert isinstance(X_, list)

    n_modes = len(X_)
    n_samples = X_[0].shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = [None] * n_modes
        for m in range(n_modes):
            ret_x[m] = X_[m][i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:upper_bound]
        yield (ret_x, ret_y)


def batch_to_feeddict(X, y, core, mode_matrices = None):
    """Prepare feed dict for session.run() from mini-batch.
    Convert sparse format into tuple (indices, values, shape) for tf.SparseTensor
    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Training vector, where batch_size in the number of samples and
        n_features is the number of features.
    y : np.array, shape (batch_size,)
        Target vector relative to X.
    core : SFMCore
        Core used for extract appropriate placeholders 
    Returns
    -------
    fd : dict
        Dict with formatted placeholders
    """
    fd = {}
    if core.isRelational and mode_matrices is not None:
        # each instance is the tuple of indicator of the mode matrix
        n_modes = len(mode_matrices)
        for m in range(n_modes):
            fd[core.train_x[m]] = X[m].astype(np.int64)
            if core.input_type == 'dense':
                fd[core.mode_matrices[m]] = mode_matrices[m].astype(np.float32)
            else:
                X_sparse = mode_matrices[m].tocoo()
                fd[core.raw_indices[m]] = np.hstack(
                    (X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis])
                ).astype(np.int64)
                fd[core.raw_values[m]] = X_sparse.data.astype(np.float32)
                fd[core.raw_shape[m]] = np.array(X_sparse.shape).astype(np.int64)
    else:
        assert isinstance(X,list)
        n_modes = len(X)
        if core.input_type == 'dense':
            for m in range(n_modes):
                fd[core.train_x[m]] = X[m].astype(np.float32)
        else:
            # sparse case
            for m in range(n_modes):
                X_sparse = X[m].tocoo()
                fd[core.raw_indices[m]] = np.hstack(
                    (X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis])
                ).astype(np.int64)
                fd[core.raw_values[m]] = X_sparse.data.astype(np.float32)
                fd[core.raw_shape[m]] = np.array(X_sparse.shape).astype(np.int64)
    if y is not None:
        fd[core.train_y] = y.astype(np.float32)
    return fd

class SFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for Structural Factorization Machines.

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.

    Parameters (for initialization)
    ----------
    view_list: list of int tuple
        # index starting from 1
        modes in each view structure, e.g., [(1,2,3),(1,4)]
        represents view1 consists of the tensor structure of mode1, mode2, mode3
        view 2 consist of the matrix of mode1 and mode4
        the number of modes is the max value in the tuple

    co_rank : int
        Number of common factors in low-rank appoximation.
        Shared by all the modes.

    view_rank: int
        Number of view-discriminative factors in low-rank appoximation.
        Shared by all the modes.

    reg_type: str
        'L1', 'L2' are supported, default: 'L2'

    reg : float, default: 0
        Strength of regularization

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.1)
        Optimization method used for training

    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    init_scaling : float, default: 2.0
        Amplitude of random initialization
        The factor augment in tf.contrib.layers.variance_scaling_initializer()
        http://www.tensorflow.org/api_docs/python/contrib.layers/initializers#variance_scaling_initializer

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching.
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .core.loss_mse, .core.loss_logistic

    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.

    Attributes
    ----------
    core : SFMCore or None
        Computational graph with internal utils.
        Will be initialized during first call .fit()

    session : tf.Session or None
        Current execution session or None.
        Should be explicitly terminated via calling destroy() method.

    steps : int
        Counter of passed lerning epochs, used as step number for writing stats

    n_feature_list : int
        Number of features in each mode used in this dataset.
        Inferred during the first call of fit() method.

    intercept : float, shape: [1]
        Intercept (bias) term.

    weights :
        list of tf.Variable, shape: [mode][view]
        List of underlying representations.
        First element in each mode will have shape [n_feature_list[mode], co_rank]
        all the others -- [n_feature_list[mode], view_rank].


    Notes
    -----
    You should explicitly call destroy() method to release resources.
    Parameter rank is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.

    References
    ----------

    """

    def init_basemodel(self, co_rank=10, view_rank=0, isFullOrder=True, view_list=None, input_type='dense', output_range = None,
                        n_epochs=100, loss_function=None, batch_size=-1, reg_type='L2', reg=0.01, init_std=0.01, init_scaling=2.0,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                        log_dir=None, session_config=None, verbose=0):
        assert view_list is not None
        self.core_arguments = {
            'co_rank': co_rank,
            'view_rank': view_rank,
            'isFullOrder': isFullOrder,
            'view_list': view_list,
            'input_type': input_type,
            'output_range': output_range,
            'loss_function': loss_function,
            'optimizer': optimizer,
            'reg_type': reg_type,
            'reg': reg,
            'init_std': init_std,
            'init_scaling': init_scaling
        }
        self.output_range = output_range
        self.core = SFMCore(**self.core_arguments)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config = session_config
        self.verbose = verbose
        self.steps = 0


    def set_core_params(self, params):
        assert isinstance(params,dict)
        for name, val in params:
            self.core_arguments[name] = val
        self.core = SFMCore(**self.core_arguments)

    def _initialize_session(self):
        """Start computational session on builded graph.

        Initialize summary logger (if needed).
        """
        if self.core.graph is None:
            raise 'Graph not found. Try call .core.build_graph() before ._initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.summary.FileWriter(
                self.log_dir,
                self.core.graph)
            if self.verbose > 0:
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(
                    os.path.abspath(self.log_dir)))
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
        gpu_options = tf.GPUOptions(allow_growth = True)
        cf = tf.ConfigProto(gpu_options = gpu_options)
        self.session = tf.Session(config= cf,
                graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    @abstractmethod
    def preprocess_target(self, target):
        """Prepare target values to use."""

    def fit(self, X_, y_, mode_matrices=None, n_epochs=None, early_stop = None, show_progress=False):
        # TODO: check this
        assert isinstance(X_,list)

        n_feature_list = [None]* len(X_)
        n_instance = X_[0].shape[0]

        if mode_matrices is not None:
            assert isinstance(mode_matrices, list)
            self.core.set_relational_input(True)
            for m, mode_matrix in enumerate(mode_matrices):
                n_feature_list[m] = mode_matrix.shape[1]
        else:
            for m, X_in_mode in enumerate(X_):
                n_feature_list[m] = X_in_mode.shape[1]

        self.core.set_num_features(n_feature_list)
        assert isinstance(self.core.n_feature_list, list)

        if self.core.graph is None:
            self.core.build_graph()
            self._initialize_session()

        used_y = self.preprocess_target(y_)

        if n_epochs is None:
            n_epochs = self.n_epochs
        
        previous_target_value = np.inf
        used_epoch = 0
        previous_core = self.core
        # Training cycle
        if self.verbose > 1:
            print('target value')
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):

            # generate permutation
            perm = np.random.permutation(n_instance)
            n_modes = len(X_)
            X_tmp = [None for i in range(n_modes)]
            for m in range(n_modes):
                X_tmp[m] = X_[m][perm]
            y_tmp = used_y[perm]
            target_value = 0
            cc = 0
            # iterate over batches
            for bX, bY in batcher(X_tmp, y_tmp, batch_size=self.batch_size):
                if self.core.isRelational:
                    fd = batch_to_feeddict(bX, bY, core=self.core, mode_matrices=mode_matrices)
                else:
                    fd = batch_to_feeddict(bX, bY, core=self.core)
                ops_to_run = [self.core.trainer, self.core.target,  self.core.summary_op]
                result = self.session.run(ops_to_run, feed_dict=fd)
#                self.session.run(self.core.post_step)
                _, batch_target_value,  summary_str = result

                target_value += batch_target_value

                # Write stats
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                self.steps += 1
                cc += 1
            if self.verbose > 1:
                print(target_value/cc)
            # warm up iterations: 100
            used_epoch = epoch
            if early_stop and epoch >= early_stop and (previous_target_value - target_value) / previous_target_value <= 1e-5:
                self.save_state('tmp/core')
                break
            previous_target_value = target_value

        return used_epoch


    def decision_function(self, X, mode_matrices=None):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        assert (self.core.isRelational and mode_matrices is not None) or \
                (not self.core.isRelational and mode_matrices is None)
        for bX, bY in batcher(X, y_=None, batch_size=self.batch_size):
            if self.core.isRelational:
                fd = batch_to_feeddict(bX, bY, core=self.core, mode_matrices=mode_matrices)
            else:
                fd = batch_to_feeddict(bX, bY, core=self.core)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        pred_y= np.concatenate(output).reshape(-1)
        # TODO: check this reshape
        return pred_y

    @abstractmethod
    def predict(self, X, mode_matrices = None):
        """Predict target values for X."""

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.b.eval(session=self.session)

    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.eval(session=self.session) for x in self.core.w]

    def save_state(self, path):
        self.core.saver.save(self.session, path)

    def load_state(self, path):
        if self.core.graph is None:
            self.core.build_graph()
            self._initialize_session()
        self.core.saver.restore(self.session, path)

    def destroy(self):
        """Terminate session and destroyes graph."""
        self.session.close()
        self.core.graph = None
