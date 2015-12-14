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

import cPickle

from scipy.stats import gmean

from nolearn.lasagne import NeuralNet


import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
def float32(k):
        return np.cast['float32'](k)

def formatData(X,y=None,Xt=None,yt=None):
        #X -= X.mean()
        #X /= X.std()
        #print X.shape
        X = X.reshape(X.shape[0],1,X.shape[1]*X.shape[2],X.shape[3])
        X = X.astype(np.float32)
        
        if y is not None:
            y = y.astype(np.int32)

        if Xt is not None:
            Xt = Xt.reshape(Xt.shape[0],1,Xt.shape[1]*Xt.shape[2],Xt.shape[3])
            Xt = Xt.astype(np.float32)
        
        if yt is not None:
            yt = yt.astype(np.int32)
        return X,y,Xt,yt

class CNN:
    def __init__(self,subject):
        self.convnet = NeuralNet(layers=[])
        self.subject = subject

    def make_cnn(self,X,y):
        #FSIZE = (int(np.floor(X.shape[2])), int(np.floor(X.shape[3]/4)))
        
        
        #FSIZE3 = (2,2)
        NUM_FILTERS1 = 16
        NUM_FILTERS2 = 32
        NUM_FILTERS3 = 256

        FSIZE1 = (X.shape[2],1)
        FSIZE2 = (NUM_FILTERS1,2)
        FSIZE3 = (NUM_FILTERS2,3)

        #x = theano.tensor.tensor4()
        #ax = theano.tensor.scalar()
        # geom_mean = theano.function(
        #     [x,axis = 3],
        #     theano.tensor.exp(theano.tensor.mean(theano.tensor.log(x), axis=axis, dtype='float32'))
        #     )
        # l2_norm = theano.function(
        #     [x,axis = 3],
        #     x.norm(2,axis=axis)
        #     )
        def geom_mean(x,axis=None):
            # x = theano.tensor.as_tensor_variable(x)
            # log = theano.tensor.log(x)
            # m = theano.tensor.mean(log,axis=axis)
            # g = m
            log = np.log(x)
            m = log.mean(axis = axis)
            g = np.exp(m)

            #g = theano.tensor.exp(m)
            #g = theano.tensor.exp(theano.tensor.mean(theano.tensor.log(x), axis=axis))
            print "gmean",g.type,g
            return g
        
        def l2_norm(x,axis=None):
            x = theano.tensor.as_tensor_variable(x)
            s = theano.tensor.sum(x,axis=axis)

            #l = x.norm(2, axis=axis)
            print "norm",l.type,l
            return l

        def me(x,axis=None):
            x = theano.tensor.as_tensor_variable(x)
            m = theano.tensor.mean(x,axis=axis)
            print "mean",m.type,m
            return m
        #print type(theano.tensor.mean),type(geom_mean),type(l2_norm)
        #learning_rate = 0.0001
        #learning_rate = 0.0005
        #learning_rate = .001
        learning_rate = .00001
        # if 'pat' in self.subject:
        #      learning_rate = 0.0001
        #FSIZE1 = (1, 2)
        #FSIZE2 = (1, X.shape[3])
        
        convnet = NeuralNet(
            layers = [
                (InputLayer,{'shape' : (None,1 , X.shape[2],X.shape[3])}),

                (Conv2DLayer,{'num_filters' : NUM_FILTERS1, 'filter_size' : FSIZE1}),

                (DropoutLayer,{'p' : .75}),
                
                (ReshapeLayer,{'shape' : ([0],[2],[1],[3])}),

                (Conv2DLayer,{'name': 'conv2', 'num_filters' : NUM_FILTERS2, 'filter_size' : FSIZE2}),

                #(DropoutLayer,{'p' : .85}),
                
                #(ReshapeLayer,{'shape' : ([0],[2],[1],[3])}),

                #(Conv2DLayer,{'name' : 'conv3', 'num_filters' : NUM_FILTERS3, 'filter_size' : FSIZE3}),
                
                (GlobalPoolLayer,{'name' : 'g1', 'incoming' : 'conv2','pool_function' : me }),
                (GlobalPoolLayer,{'name' : 'g2', 'incoming' : 'conv2','pool_function' : theano.tensor.max }),
                (GlobalPoolLayer,{'name' : 'g3', 'incoming' : 'conv2','pool_function' : theano.tensor.min }),
                (GlobalPoolLayer,{'name' : 'g4', 'incoming' : 'conv2','pool_function' : theano.tensor.var }),
                #(GlobalPoolLayer,{'name' : 'g5', 'incoming' : 'conv2','pool_function' : geom_mean}),
                #(GlobalPoolLayer,{'name' : 'g6', 'incoming' : 'conv2','pool_function' : l2_norm }),
                
                (ConcatLayer,{'incomings' : ['g1','g2','g3','g4']}),#]}),#
                
                (DenseLayer, {'num_units': 256}),
                (DropoutLayer,{'p':.5}),
                (DenseLayer, {'num_units': 256}),

                (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
            ],

            update_learning_rate=theano.shared(float32(learning_rate)),
            update_momentum=theano.shared(float32(0.9)),
            verbose=1,
            max_epochs = 100000,
            on_epoch_finished=[
            EarlyStopping(patience=100)
            ],
            )
        return convnet

    def fit(self,X,y,xt,yt):
        
        X,y,xt,yt = formatData(X,y=y,Xt=xt,yt=yt)
        self.convnet = self.make_cnn(X,y)
        print "shape",X.shape
        self.convnet.fit(X,y,xt,yt)
        

    def predict_proba(self,X):
        X,_,_,_ = formatData(X)
        return self.convnet.predict_proba(X)

    def predict(self,X):
        X,_,_,_ = formatData(X)
        return self.convnet.predict(X)

    def get_params(self,deep):
        return self.convnet.get_params()
    def load_params_from(self,net):
        return self.convnet.load_params_from(net)


