import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.nonlinearities import softmax
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


class CNN:
    def __init__(self):
        self.convnet = None

    def make_cnn(self,X,y):
        FSIZE = (int(np.floor(X.shape[2]/4)), int(np.floor(X.shape[3]/4)))
        #FSIZE = (2,8)
        NUM_FILTERS1 = 8
        NUM_FILTERS2 = 16
        FSIZE1 = (X.shape[2], 1)
        FSIZE2 = (1, X.shape[3])
        
        convnet = NeuralNet(
            layers = [
                ('input',InputLayer ),

                ('conv1',Conv2DLayer),
                (DropoutLayer,{'p':.5}),
                ('conv2',Conv2DLayer),

                ('g1',GlobalPoolLayer),
                ('g2',GlobalPoolLayer),
                ('g3',GlobalPoolLayer),
                ('g4',GlobalPoolLayer),
                
                ('concat',ConcatLayer),
                
                (DenseLayer, {'num_units': 128}),
                (DropoutLayer,{'p':.5}),
                (DenseLayer, {'num_units': 128}),

                (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
            ],
            
            input_shape= (None, 1, X.shape[2],X.shape[3]),
            
            conv1_num_filters = NUM_FILTERS1,
            conv1_filter_size = FSIZE , 
            conv1_pad =  1,

            conv2_num_filters = NUM_FILTERS2,
            conv2_filter_size = FSIZE , 
            conv2_pad =  1,

            g1_incoming = 'conv2',
            g2_incoming = 'conv2',
            g3_incoming = 'conv2',
            g4_incoming = 'conv2',
            
            g1_pool_function = theano.tensor.mean,
            g2_pool_function = theano.tensor.max,
            g3_pool_function = theano.tensor.min,
            g4_pool_function = theano.tensor.var,
            
            concat_incomings = ['g1','g2','g3','g4'],

            update_learning_rate=theano.shared(float32(0.01)),
            update_momentum=theano.shared(float32(0.9)),
            verbose=2,
            max_epochs = 50,

            )
        return convnet

    def fit(self,X,y):
        X,y = formatData(X,y)
        print X.shape
        self.convnet = self.make_cnn(X,y)
        self.convnet.fit(X,y)

    def predict_proba(self,X):
        X,_ = formatData(X)
        return self.convnet.predict_proba(X)

    def predict(self,X):
        X,_ = formatData(X)
        return self.convnet.predict(X)
