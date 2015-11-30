import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import MaxPool2DLayer
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
        convnet = NeuralNet(
            layers = [
                (InputLayer, {'shape': (None, 1, X.shape[2],X.shape[3])}),

                (Conv2DLayer, {'num_filters': 8, 'filter_size': FSIZE , 'pad': 1}),
                
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':.5}),

                (Conv2DLayer, {'num_filters': 16, 'filter_size': FSIZE, 'pad': 1}),
                
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':.5}),

                (DenseLayer, {'num_units': 256}),
                (DropoutLayer, {'p':.5}),
                (DenseLayer, {'num_units': 256}),

                (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
            ],
            update_learning_rate=theano.shared(float32(0.005)),
            update_momentum=theano.shared(float32(0.9)),
            verbose=2,
            max_epochs = 10,
            )
        return convnet

    def fit(self,X,y):
        X,y = formatData(X,y)
        self.convnet = self.make_cnn(X,y)
        self.convnet.fit(X,y)

    def predict_proba(self,X):
        X,_ = formatData(X)
        return self.convnet.predict_proba(X)

