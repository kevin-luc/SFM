"""
    Implementation of the core functions for the Structural Factorization Machines in TensorFlow
"""
from __future__ import (absolute_import, division,
                                print_function, unicode_literals)
#from builtins import *
import tensorflow as tf
# from . import utils
import math


class SFMCore():
    """
    This class underlying routine about creating computational graph.
    Its required n_features to be set at graph building time.

    Parameters
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

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.1)
        Optimization method used for training

    reg_type: str
        'L1', 'L2', 'L21', 'maxNorm' are supported, default: 'L2'

    reg : float, default: 0
        Strength of regularization

    init_scaling : float, default: 2.0
        Amplitude of random initialization
        The factor augment in tf.contrib.layers.variance_scaling_initializer()
        http://www.tensorflow.org/api_docs/python/contrib.layers/initializers#variance_scaling_initializer

    Attributes
    ----------
    graph : tf.Graph or None
        Initialized computational graph or None

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    n_feature_list : list of int
        Number of features in each mode used in this dataset.
        Inferred during the first call of fit() method.

    saver : tf.Op
        tf.train.Saver instance, connected to graph

    summary_op : tf.Op
        tf.merge_all_summaries instance for export logging

    b : list of tf.Variable, shape: [mode]
        Bias term for each mode

    W : list of tf.Variable, shape: [n_mode][n_view]
        List of underlying representations.
        First element in each mode will have shape [n_feature_list[mode], co_rank]
        all the others -- [n_feature_list[mode], view_rank].

    Notes
    -----
    
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.
    This implementation uses a generalized approach from referenced paper along
    with caching.

    References
    ----------

    """
    def __init__(self, view_list, co_rank, view_rank, isFullOrder, input_type, output_range,
                    loss_function, optimizer, reg_type, reg, init_std, init_scaling):
        self.view_list = view_list
        self.co_rank = co_rank
        self.view_rank = view_rank
        self.input_type = input_type
        self.output_range = output_range
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.reg = reg
        self.init_std = init_std
        self.init_scaling = init_scaling
        self.n_modes = max([x for v in view_list for x in v ])
        self.n_views = len(view_list)
        self.n_feature_list = None
        self.mode_matrices = None
        self.graph = None
        self.isRelational = False
        self.isFullOrder = isFullOrder

    def set_relational_input(self, isRelational):
        assert isinstance(isRelational,bool)
        self.isRelational = isRelational

    def set_num_features(self, n_feature_list):
        self.n_feature_list = n_feature_list

    def _init_learnable_params(self):
        self.W = [[None] * self.n_modes for i in range(self.n_views + 1)]
        self.Bias = [[None] * self.n_modes for i in range(self.n_views + 1)]
        self.S = [None] * self.n_modes
        r = self.co_rank + self.view_rank

#        if self.reg_type == 'L1':
            #regular = tf.contrib.layers.l1_regularizer(self.reg)
        #else:
            #regular = tf.contrib.layers.l2_regularizer(self.reg)

        # initialize factors for each view
        # to avoid the multiplication close to zero when the views has more than 3 modes
        # we can try to scaling the unfiorm distribution using variance_scaling_initializer

        self.Phi = tf.get_variable('embedding_phi', shape = [r, self.n_views], trainable=True,
                                    initializer = tf.contrib.layers.variance_scaling_initializer(factor = self.init_scaling))

        self.b = tf.Variable(0.0, trainable=True, name='b')
        # initialize shared factors for each mode
        for m in range(self.n_modes):
            with tf.variable_scope('co_mode_'+str(m+1)):
                self.W[0][m] = tf.get_variable('embedding_init',
                           shape = [self.n_feature_list[m], self.co_rank],
                           trainable=True,
                           initializer = tf.contrib.layers.variance_scaling_initializer(factor = self.init_scaling))
                self.S[m] = tf.get_variable('layer_norm_S', initializer = tf.ones([r]))

        # initialize view specific facotrs for each mode
        for i, modes in enumerate(self.view_list):
            v = i+1
            for m in set(modes):
                with tf.variable_scope('view_'+str(v)+'_mode_' + str(m)):
                    try:
                        self.Bias[v][m-1] = tf.get_variable('bias', shape = [1,r],
                                trainable = self.isFullOrder,
                                initializer=tf.zeros_initializer())
                    except:
                        print('bias mode {} shared in view {}'.format(m,v))
                    try:
                        if self.view_rank>0:
                            self.W[v][m-1] = tf.get_variable('embedding_init',
                                shape = [self.n_feature_list[m-1], self.view_rank],
                                trainable=True,
                                initializer = tf.contrib.layers.variance_scaling_initializer(factor = init_scaling))
                    except:
                        print('mode {} shared in view {}'.format(m,v))


    def _init_placeholders(self):
        self.train_x = [None]*self.n_modes
        if self.isRelational:
            self.mode_matrices = [None] * self.n_modes

        #sparse case
        if self.input_type != 'dense':
            self.raw_indices = [None]*self.n_modes
            self.raw_values = [None]*self.n_modes
            self.raw_shape = [None]*self.n_modes

        for i in range(self.n_modes):
            with tf.variable_scope('mode_'+str(i+1)):
                # if given mode matrix, the input X_ is the list of the row indicators of the mode matrix
                if self.input_type == 'dense':
                    if self.isRelational:
                        self.train_x[i] = tf.placeholder(tf.int64, shape=[None], name='X_indices')
                        self.mode_matrices[i] = tf.placeholder(tf.float32, shape=[None, self.n_feature_list[i]], name='X_matrix')
                    else:
                        self.train_x[i] = tf.placeholder(tf.float32, shape=[None, self.n_feature_list[i]], name='X')
                else:
                    #sparse case
                    self.raw_indices[i] = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                    self.raw_values[i] = tf.placeholder(tf.float32, shape=[None], name='raw_data')
                    self.raw_shape[i] = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
                    if self.isRelational:
                        self.train_x[i] = tf.placeholder(tf.int64, shape=[None], name='X_indices')
                        self.mode_matrices[i] = tf.SparseTensor(self.raw_indices[i], self.raw_values[i], self.raw_shape[i])
                    # tf.sparse_reorder is not needed since scipy return COO in canonical order
                    else:
                        self.train_x[i] = tf.SparseTensor(self.raw_indices[i], self.raw_values[i], self.raw_shape[i])
        self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')
    def _batch_norm(self, Z, s, b):
        eps = 1e-5
        # Calculate batch mean and variance
        m, v = tf.nn.moments(Z, [0], keep_dims = True)

        # Apply the initial batch normalizing transform
        normalized_Z = (Z - m) / tf.sqrt(v + eps)
        return normalized_Z * s + b

    def _layer_norm(self, Z, s, b):
        eps = 1e-5
        m, v = tf.nn.moments(Z, [1], keep_dims = True)
        normalized_Z = (Z - m) / tf.sqrt(v + eps)
        return normalized_Z * s + b

    def _regularizer_func(self, W, node_name):
        if self.reg_type == 'L1':
            norm = tf.reduce_sum(tf.abs(W), name=node_name)
        else:
            norm = tf.nn.l2_loss(W, name=node_name)
        return norm

    def _init_regular(self):
        self.regularization = 0
        tf.summary.scalar('bias', self.b)

        self.regularization = 0
        for m in range(self.n_modes):
            node_name = 'regularization_penalty_v0_m{}'.format(m)
            norm = self._regularizer_func(self.W[0][m],node_name)
            tf.summary.scalar('norm_W_v0_m{}'.format(m), norm)
            self.regularization += norm
        for i, modes in enumerate(self.view_list):
            v = i + 1
            for m in set(modes):
                try:
                    node_name = 'regularization_penalty_v{}_b{}'.format(v, m)
                    norm = self._regularizer_func(self.Bias[v][m-1], node_name)
                    tf.summary.scalar('norm_Bias_v{}_m{}'.format(v,m), norm)
                except:
                    print('bias mode {} shared in view {}'.format(m,v))
                self.regularization += norm
                if self.view_rank > 0:
                    try:
                        node_name = 'regularization_penalty_v{}_m{}'.format(v,m)
                        norm = self._regularizer_func(self.W[v][m-1],node_name)
                        tf.summary.scalar('norm_W_v{}_m{}'.format(v,m), norm)
                    except:
                        print('mode {} shared in view {}'.format(m,v))
                    self.regularization += norm

        for v in range(len(self.view_list)):
            norm = self._regularizer_func(self.Phi[:,v], 'regularization_penalty_phi{}'.format(v+1))
            tf.summary.scalar('norm_Phi_v{}'.format(v+1), norm)
        node_name = 'regularization_penalty_phi'
        norm = self._regularizer_func(self.Phi, node_name)
        self.regularization += norm
        tf.summary.scalar('regularization_penalty', self.regularization)

    def _init_loss(self):
        self.loss = self.loss_function(self.outputs, self.train_y)
        self.reduced_loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.reduced_loss)

    def _init_main_block(self):
        self.prod_view = {}
        r = self.co_rank + self.view_rank

#        self.outputs = self.b
        self.outputs = 0
        train_shape = [tf.shape(self.train_x[0])[0], r]
        self.XW_cache = {}
        self.prod_embedding = [None] * self.n_views
        self.view_contribution = [None] * self.n_views

        for m in range(self.n_modes):
            self.XW_cache[m] = self._view_mode_embedding(0, m)

        for i, modes in enumerate(self.view_list):
            v = i + 1
            with tf.name_scope('view_{}'.format(v)) as scope:
                XW_list = [None] * self.n_modes
                # noting that the modes given in the input start from 1
                for m in modes:
                    with tf.name_scope('mode_{}'.format(m)) as scope:
                        if self.view_rank > 0:
                            XW = self._view_mode_embedding(v, m - 1)
                            XW_list[m-1] = tf.concat(axis=1, values=[self.XW_cache[m-1], XW], name='XW')
                        else:
                            XW_list[m-1] = self.XW_cache[m-1]
                        XW_list[m-1] += self.Bias[v][m-1]
#                        XW_list[m-1] = self._batch_norm(XW_list[m-1], self.S[m-1], self.Bias[v][m-1])
#                        XW_list[m-1] = self._layer_norm(XW_list[m-1], self.S[m-1], self.Bias[v][m-1])

                # the reduction_indices in the reduce_prod does not handle scalar, 
                # so we need to transform it to a tensor
                embedding_tensor = tf.stack([xw for xw in XW_list if xw is not None],axis=2, name='embedding_tensor')
                self.prod_embedding[i] = tf.reduce_prod(embedding_tensor, axis=[2], name='prod_embedding')

                self.view_contribution[i] = matmul_wrapper(self.prod_embedding[i], tf.reshape(self.Phi[:,i],(r,1)), 'dense')
                tf.summary.histogram('view_contribution{}'.format(v), self.view_contribution[i])

        self.outputs += tf.reduce_sum(self.view_contribution, axis=[0], name='output')
        tf.summary.histogram('output', self.outputs)

        with tf.name_scope('loss') as scope:
            self._init_loss()

        with tf.name_scope('regularization') as scope:
            self._init_regular()

    def _view_mode_embedding(self, v, m):
        if self.isRelational:
            modeEmbedding = matmul_wrapper(self.mode_matrices[m], self.W[v][m], self.input_type)
            XW = tf.nn.embedding_lookup(modeEmbedding, self.train_x[m])
        else:
            XW = matmul_wrapper(self.train_x[m], self.W[v][m], self.input_type)
        return XW


    def _init_target(self):
#        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.target = self.reduced_loss + self.reg * self.regularization

        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value', name='target')
        tf.summary.scalar('target', self.checked_target)

    def build_graph(self):
        """Build computational graph according to params."""
        assert self.n_feature_list is not None
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('params') as scope:
                self._init_learnable_params()

            with tf.name_scope('inputBlock') as scope:
                self._init_placeholders()

            with tf.name_scope('mainBlock') as scope:
                self._init_main_block()

            self._init_target()

            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init_all_vars = tf.global_variables_initializer()
#            self.post_step = self._norm_constraint_op()
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

def matmul_wrapper(A, B, optype):
    """Wrapper for handling sparse and dense versions of matmul operation.

    Parameters
    ----------
    A : tf.Tensor
    B : tf.Tensor
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """

    if optype == 'dense':
        return tf.matmul(A, B)
    elif optype == 'sparse':
        return tf.sparse_tensor_dense_matmul(A, B)
    else:
        raise NameError('Unknown input type in matmul_wrapper')

#def L2Ball_update(var_matrix, maxnorm=1.0):
    #'''Dense update operation that ensures all columns in var_matrix 
        #have a Euclidean norm equal to maxnorm. 

    #Args:
        #var_matrix: 2D mutable tensor (Variable) to operate on
        #maxnorm: the maximum Euclidean norm
        
    #Returns:
        #An operation that will update var_matrix when run in a Session
    #'''
    #scaling = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), 0))
    #scaled = var_matrix / tf.expand_dims(scaling, 1)
    #return tf.assign(var_matrix, scaled)


#def L2Ball(var_matrix, maxnorm=1.0):
    #'''Similar to L2Ball_update(), except this returns a new Tensor
       #instead of an operation that modifies var_matrix.

    #Args:
        #var_matrix: 2D tensor (Variable)
        #maxnorm: the maximum Euclidean norm

    #Returns:
        #A new tensor where all rows have been scaled as necessary
    #'''
    #scaling = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), 0))
    #return var_matrix / tf.expand_dims(scaling, 0)

#def pow_wrapper(X, p, optype):
    #"""Wrapper for handling sparse and dense versions of power operation.

    #Parameters
    #----------
    #X : tf.Tensor
    #p : int
    #optype : str, {'dense', 'sparse'}

    #Returns
    #-------
    #tf.Tensor
    #"""
    #if optype == 'dense':
        #return tf.pow(X, p)
    #elif optype == 'sparse':
        #return tf.SparseTensor(X.indices, tf.pow(X.values, p), X.shape)
    #else:
        #raise NameError('Unknown input type in pow_wrapper')
