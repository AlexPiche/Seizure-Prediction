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
        self.convnet = None

    def make_rnn(self,X,y):
        FSIZE = (int(np.floor(X.shape[2]/4)), int(np.floor(X.shape[3]/4)))

        recnet = NeuralNet(
            layers = [
                ('input', InputLayer ),

                ('lstm_forward', LSTMLayer),
                ('lstm_backward', LSTMLayer),

                ('concat', ConcatLayer),

                ('lstm_sum', ElementwiseSumLayer),

                (Reshape),

                (DenseLayer, {'num_units': 1, 'nonlinearity': tanh}),
            ],

            input_shape= (None, 1, X.shape[2], X.shape[3]),

            conv1_num_filters = NUM_FILTERS1,
            conv1_filter_size = FSIZE , 
            conv1_pad =  1,

            conv2_num_filters = NUM_FILTERS2,
            conv2_filter_size = FSIZE , 
            conv2_pad =  1,

            lstm_forward_incoming = 'input',
            lstm_backward_incoming = 'input',
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
        self.recnet = self.make_rnn(X,y)
        self.recnet.fit(X,y)

    def predict_proba(self,X):
        X,_ = formatData(X)
        return self.recnet.predict_proba(X)

    def predict(self,X):
        X,_ = formatData(X)
        return self.recnet.predict(X)
