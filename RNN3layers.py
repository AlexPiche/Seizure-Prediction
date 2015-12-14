import theano
import sklearn
import os
from sklearn.cross_validation import train_test_split
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
import numpy as np
import UtilitiesRNN

def binary_accuracy(predictions, targets, threshold=0.5):
    """Computes the binary accuracy between predictions and targets.
    .. math:: L_i = \\mathbb{I}(t_i = \mathbb{I}(p_i \\ge \\alpha))
    Parameters
    ----------
    predictions : Theano tensor
        Predictions in [0, 1], such as a sigmoidal output of a neural network,
        giving the probability of the positive class
    targets : Theano tensor
        Targets in {0, 1}, such as ground truth labels.
    threshold : scalar, default: 0.5
        Specifies at what threshold to consider the predictions being of the
        positive class
    Returns
    -------
    Theano tensor
        An expression for the element-wise binary accuracy in {0, 1}
    Notes
    -----
    This objective function should not be used with a gradient calculation;
    its gradient is zero everywhere. It is intended as a convenience for
    validation and testing, not training.
    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    predictions = theano.tensor.ge(predictions, threshold)
    return theano.tensor.eq(predictions, targets)

#target_values = T.vector('target_output')
target_values = T.ivector('target_output')
mask = T.matrix('mask')

# Recurrent Networks

class RNN:
    def __init__(self, flavour='gru', N_HIDDEN=50, N_HIDDEN2=100, N_HIDDEN3=100, NUM_EPOCHS=150, patient=None):
        self.rnn = None
        self.__N_HIDDEN = N_HIDDEN
        self.__N_HIDDEN2 = N_HIDDEN2
        self.__N_HIDDEN3 = N_HIDDEN3
        self.__flavour = flavour
        self.__NUM_EPOCHS = NUM_EPOCHS
        global file_title
        file_title = "analysis/" + str(flavour) + str(N_HIDDEN) + str(patient) + '.csv'
    def make_rnn(self, X, y, op1=T.minimum, op2=T.minimum, op3=T.minimum, peepholes=True):
        """
        Build a RNN

        flavour can be 'lstm', 'gru', or vanilla
        operation can be T.add, T.mul, T.minimum, or T.maximum
        """
        N_HIDDEN = self.__N_HIDDEN
        N_HIDDEN2 = self.__N_HIDDEN2
        N_HIDDEN3 = self.__N_HIDDEN3
        flavour = self.__flavour


        rnn = {}
        rnn['input'] = lasagne.layers.InputLayer(shape=(None, None, X.shape[2]))

        rnn['mask'] = lasagne.layers.InputLayer(shape=(None, None))

        gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))

        cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            W_cell=None, b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh)

        rnn['dropout1'] = lasagne.layers.DropoutLayer(rnn['input'], p=0.5)
        rnn['noise'] = lasagne.layers.GaussianNoiseLayer(rnn['dropout1'], sigma=1)

        # Bidirectional RNN
        # which kind of gate

        if flavour == 'lstm' :
            print "Initializing LSTM"
            rnn['rnn_forward'] = lasagne.layers.recurrent.LSTMLayer(
                rnn['noise'], N_HIDDEN, mask_input=rnn['mask'],
                ingate=gate_parameters, forgetgate=gate_parameters,
                cell=cell_parameters, outgate=gate_parameters,
                learn_init=True, grad_clipping=100.,
                peepholes=peepholes)

            rnn['rnn_backward'] = lasagne.layers.recurrent.LSTMLayer(
                rnn['noise'], N_HIDDEN, ingate=gate_parameters,
                mask_input=rnn['mask'], forgetgate=gate_parameters,
                cell=cell_parameters, outgate=gate_parameters,
                learn_init=True, grad_clipping=100., backwards=True,
                peepholes=peepholes)

        elif flavour == 'gru' :
            print "Initializing GRU"
            rnn['rnn_forward'] = lasagne.layers.recurrent.GRULayer(
            rnn['noise'], N_HIDDEN, mask_input=rnn['mask'],
            resetgate=gate_parameters , updategate=gate_parameters,
                learn_init=True, grad_clipping=100.)

            rnn['rnn_backward'] = lasagne.layers.recurrent.GRULayer(
            rnn['noise'], N_HIDDEN, mask_input=rnn['mask'],
	    resetgate=gate_parameters , updategate=gate_parameters,
            learn_init=True, grad_clipping=100., backwards=True)

        else:
            print "Initializing Vanilla Recurrent Layer"
            rnn['rnn_forward'] = lasagne.layers.RecurrentLayer(rnn['noise'],
            N_HIDDEN, mask_input=rnn['mask'],
            nonlinearity = lasagne.nonlinearities.tanh,
            learn_init=True, grad_clipping=100., backwards=False)

            rnn['rnn_backward'] = lasagne.layers.RecurrentLayer(rnn['noise'],
            N_HIDDEN, mask_input=rnn['mask'],
            nonlinearity = lasagne.nonlinearities.tanh,
            learn_init=True, grad_clipping=100., backwards=True)

        #rnn['merge'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward'], rnn['rnn_backward']], T.minimum)

        if N_HIDDEN2 > 0:
            rnn['noise2'] = lasagne.layers.GaussianNoiseLayer(rnn['rnn_forward'], sigma=1)
            rnn['dropout2'] = lasagne.layers.DropoutLayer(rnn['noise2'], p=0.5)

            rnn['rnn_forward2'] = lasagne.layers.recurrent.GRULayer(
				rnn['dropout2'], N_HIDDEN2, mask_input=rnn['mask'],
				resetgate=gate_parameters , updategate=gate_parameters,
				learn_init=True, grad_clipping=100.)

            rnn['rnn_backward2'] = lasagne.layers.recurrent.GRULayer(
				rnn['dropout2'], N_HIDDEN2, mask_input=rnn['mask'],
				resetgate=gate_parameters , updategate=gate_parameters,
				learn_init=True, grad_clipping=100., backwards=True)

            rnn['merge2'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward2'], rnn['rnn_backward2']], op1)

            rnn['noise3'] = lasagne.layers.GaussianNoiseLayer(rnn['rnn_backward'], sigma=1)
            rnn['dropout3'] = lasagne.layers.DropoutLayer(rnn['noise3'], p=0.5)

            rnn['rnn_forward3'] = lasagne.layers.recurrent.GRULayer(
				rnn['dropout3'], N_HIDDEN2, mask_input=rnn['mask'],
				resetgate=gate_parameters , updategate=gate_parameters,
				learn_init=True, grad_clipping=100.)

            rnn['rnn_backward3'] = lasagne.layers.recurrent.GRULayer(
				rnn['dropout3'], N_HIDDEN2, mask_input=rnn['mask'],
				resetgate=gate_parameters , updategate=gate_parameters,
				learn_init=True, grad_clipping=100., backwards=True)

            rnn['merge3'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward3'], rnn['rnn_backward2']], op1)

            if N_HIDDEN3 > 0:

                rnn['merge4'] = lasagne.layers.ElemwiseMergeLayer([rnn['merge2'], rnn['merge3']], op2)
                rnn['noise4'] = lasagne.layers.GaussianNoiseLayer(rnn['merge4'], sigma=1)
                rnn['dropout4'] = lasagne.layers.DropoutLayer(rnn['noise4'], p=0.5)

                rnn['rnn_forward4'] = lasagne.layers.recurrent.GRULayer(
                    rnn['dropout4'], N_HIDDEN3, mask_input=rnn['mask'],
                    resetgate=gate_parameters , updategate=gate_parameters,
                    learn_init=True, grad_clipping=100.)

                rnn['rnn_backward4'] = lasagne.layers.recurrent.GRULayer(
                    rnn['dropout4'], N_HIDDEN3, mask_input=rnn['mask'],
                    resetgate=gate_parameters , updategate=gate_parameters,
                    learn_init=True, grad_clipping=100., backwards=True)

                rnn['merge5'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward4'], rnn['rnn_backward4']], op3)
                rnn['noise5'] = lasagne.layers.GaussianNoiseLayer(rnn['merge5'], sigma=1)
                rnn['dropout5'] = lasagne.layers.DropoutLayer(rnn['noise5'], p=0.5)

                rnn['reshape'] = ReshapeLayer(rnn['dropout5'], (-1, N_HIDDEN3))


        rnn['dense'] = DenseLayer(rnn['reshape'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        batchsize, seqlen, _ = rnn['input'].input_var.shape
        rnn['out'] = ReshapeLayer(rnn['dense'], (batchsize, seqlen)) 
        return rnn

    def fit(self, X, y, valid_set_x, valid_set_y, key=0, EPOCH_SIZE=1, max_patience=15):
        #with open(file_title, "a") as myfile:
        #            myfile.write('\n')
        #            myfile.write(str(key))
        NUM_EPOCHS=self.__NUM_EPOCHS
        self.rnn = self.make_rnn(X, y)
        # initiate best loss value to 0 correct prediction
        best_val = 0
        #initialize patience to 1
        patience = 1
        network_output = lasagne.layers.get_output(self.rnn['out'])
        training_output = lasagne.layers.get_output(self.rnn['out'],deterministic=True)
        predicted_values = network_output[:, -1]
        pred_validation = training_output[:, -1]
        l2_penalty = regularize_layer_params(self.rnn['out'], l2) * 1e-4
        cost = lasagne.objectives.binary_crossentropy(predicted_values, target_values).mean() + l2_penalty
        acc_val = aggregate(binary_accuracy(pred_validation, target_values, threshold=0.5))
        all_params = lasagne.layers.get_all_params(self.rnn['out'])#, trainable = True)
        #updates = lasagne.updates.adam(cost, all_params)
        #updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate=0.01, momentum=0.9)
        updates = lasagne.updates.rmsprop(cost, all_params, 0.01)
        train = theano.function(
            [self.rnn['input'].input_var, target_values, self.rnn['mask'].input_var],
            cost, updates=updates, allow_input_downcast=True)
        compute_cost = theano.function(
            [self.rnn['input'].input_var, target_values, self.rnn['mask'].input_var],
            cost, allow_input_downcast=True)
        compute_acc = theano.function(
            [self.rnn['input'].input_var, target_values, self.rnn['mask'].input_var],
            acc_val, allow_input_downcast=True)
        mask_train = np.ones((X.shape[0], X.shape[2]))
        mask_valid = np.ones((valid_set_x.shape[0], valid_set_x.shape[2]))
        print("Beginning training")
        for epoch in range(NUM_EPOCHS):
            for _ in range(EPOCH_SIZE):
                cost = train(X, y, mask_train)
            #acc_val = compute_acc(valid_set_x, valid_set_y, mask_valid)
            yhat = self.predict_proba(valid_set_x)
            acc_val = sklearn.metrics.roc_auc_score(valid_set_y, yhat)
            #with open(file_title, "a") as myfile:
            #        myfile.write(','+str(acc_val))
            if acc_val > best_val:
                patience = 1
                best_val = acc_val
                best_model = lasagne.layers.get_all_param_values(self.rnn['out'])
            else:
                patience += 1
            if patience > max_patience:
                break
        lasagne.layers.set_all_param_values(self.rnn['out'], best_model)
        print("Epoch {} Best validation accuracy = {}".format(epoch + 1, best_val))
        best_yhat = self.predict_proba(valid_set_x)
        return best_val, best_yhat

    def predict_proba(self, X):
        mask_predict = np.ones((X.shape[0], X.shape[2]))
        network_output   = get_output(self.rnn['out'], deterministic=True)
        predicted_values = network_output[:, -1]
        predict = theano.function([self.rnn['input'].input_var, self.rnn['mask'].input_var], predicted_values, allow_input_downcast=True)
        prediction = predict(X, mask_predict)
        return prediction

