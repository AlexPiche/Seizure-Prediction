import pdb
import lasagne
from lasagne.layers import *
from lasagne.layers.recurrent import *
from lasagne.nonlinearities import tanh
from lasagne.updates import adam
from lasagne.layers import get_all_params
from lasagne.updates import nesterov_momentum
from sklearn.utils import shuffle
import theano


from nolearn.lasagne import NeuralNet


import numpy as np

gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))

cell_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
    W_cell=None, b=lasagne.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
    nonlinearity=lasagne.nonlinearities.tanh)

#l_in = lasagne.layers.InputLayer(shape=(None, None, N_FEATURES_DIM))
#l_mask = lasagne.layers.InputLayer(shape=(None, None))

# LSTM number of hidden/cell units
N_HIDDEN = 10

def float32(k):
        return np.cast['float32'](k)

def formatData(X, y = None):
        #X -= X.mean()
        #X /= X.std()
        X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
        X = X.astype(np.float32)
        if y is not None:
                y = y.astype(np.int32)
        return X,y


class RNN:
    def __init__(self):
            self.recnet = None

    def make_rnn(self,X,y):
            recnet = NeuralNet(
                    layers = [
                            ('input', InputLayer ),
                            ('lstm_forward', LSTMLayer),
                            ('lstm_backward', LSTMLayer, {'backwards': True}),
                            ('concat', ConcatLayer),
                            ('lstm_sum', ElemwiseSumLayer),
                            (ReshapeLayer, {"shape": (-1, N_HIDDEN)}),
                            (DenseLayer, {'num_units': 1, 'nonlinearity': tanh}),
                    ],

                    input_shape= (None, None, X.shape[2]),

                    # LSTM parameters
                    lstm_forward_incoming = 'input',
                    lstm_forward_num_units = N_HIDDEN,
                    lstm_forward_ingate=gate_parameters,
                    lstm_forward_forgetgate=gate_parameters,
                    lstm_forward_cell=cell_parameters,
                    lstm_forward_outgate=gate_parameters,
                    lstm_forward_learn_init=True,
                    lstm_forward_grad_clipping=100.0,
                    lstm_backward_incoming = 'input',
                    lstm_backward_num_units = N_HIDDEN,
                    lstm_backward_ingate=gate_parameters,
                    lstm_backward_backgetgate=gate_parameters,
                    lstm_backward_cell=cell_parameters,
                    lstm_backward_outgate=gate_parameters,
                    lstm_backward_learn_init=True,
                    lstm_backward_grad_clipping=100.0,
                    #lstm_backward_backwards=True,

                    concat_incomings = ['lstm_forward', 'lstm_backward'],
                    lstm_sum_incoming = 'concat',

                    update_learning_rate=theano.shared(float32(0.01)),
                    update_momentum=theano.shared(float32(0.9)),
                    verbose=2,
                    max_epochs = 50,
            )
            return recnet

    def fit(self,X,y):
            X,y = formatData(X,y)
            print X.shape
            print y.shape
            self.recnet = self.make_rnn(X,y)
            self.recnet.fit(X,y)

    def predict_proba(self,X):
            X,_ = formatData(X)
            return self.recnet.predict_proba(X)

    def predict(self,X):
            X,_ = formatData(X)
            return self.recnet.predict(X)
