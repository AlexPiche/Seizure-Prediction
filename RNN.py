import theano
from sklearn.cross_validation import train_test_split
import theano.tensor as T
import lasagne
import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt
# Symbolic variable for the target network output.
# It will be of shape n_batch, because there's only 1 target value per sequence.
target_values = T.vector('target_output')
# This matrix will tell the network the length of each sequences.
# The actual values will be supplied by the gen_data function.
mask = T.matrix('mask')

# Recurrent Networks

class RNN:
    def __init__(self):
        self.rnn = None
    def make_rnn(self, X, y, N_HIDDEN = 50, flavour = 'lstm', operation = T.add):
        """
        Build a RNN

        flavour can be 'lstm', 'gru', or vanilla
        operation can be T.add, T.mul, T.minimum, or T.maximum
        """

        print "Building the Model"
        rnn = {}
        N_FEATURES_DIM = X.shape[2]
        rnn['input'] = lasagne.layers.InputLayer(shape=(None, None, N_FEATURES_DIM))
        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (n_batch, n_time_steps);
        # both of these dimensions are variable for us so we will use
        # an input shape of (None, None)
        rnn['mask'] = lasagne.layers.InputLayer(shape=(None, None))
        # All gates have initializers for the input-to-gate and hidden state-to-gate
        # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
        # The convention is that gates use the standard sigmoid nonlinearity,
        # which is the default for the Gate class.
        gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))

        cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            # Setting W_cell to None denotes that no cell connection will be used.
            W_cell=None, b=lasagne.init.Constant(0.),
            # By convention, the cell nonlinearity is tanh in an RNN.
            nonlinearity=lasagne.nonlinearities.tanh)

        # Add noise to the input
        rnn['noise'] = lasagne.layers.GaussianNoiseLayer(rnn['input'], sigma=0.1)

        # Bidirectional RNN

        if flavour == 'lstm' :
            print "Initializing LSTM"
            rnn['rnn_forward'] = lasagne.layers.recurrent.LSTMLayer(
            rnn['noise'], N_HIDDEN,
            mask_input=rnn['mask'],
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=100.)

            rnn['rnn_backward'] = lasagne.layers.recurrent.LSTMLayer(
            rnn['noise'], N_HIDDEN, ingate=gate_parameters,
            mask_input=rnn['mask'], forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=100., backwards=True)

        elif flavour == 'gru' :
            print "Initializing GRU"
            rnn['rnn_forward'] = lasagne.layers.recurrent.GRULayer(
            rnn['noise'], N_HIDDEN, mask_input=rnn['mask'],
            resetgate=gate_parameters , updategate=gate_parameters,
            learn_init=True, grad_clipping=100.)

            rnn['rnn_backward'] = lasagne.layers.recurrent.GRULayer(
            rnn['noise'], N_HIDDEN, mask_input=rnn['mask'],
            learn_init=True, grad_clipping=100., backwards=True)

        else:
            print "Initializing Recurrent Layer"
            rnn['rnn_forward'] = lasagne.layers.RecurrentLayer(rnn['noise'],
            N_HIDDEN, mask_input=rnn['mask'],
            nonlinearity = lasagne.nonlinearities.tanh,
            learn_init=True, grad_clipping=100., backwards=False)

            rnn['rnn_backward'] = lasagne.layers.RecurrentLayer(rnn['noise'],
            N_HIDDEN, mask_input=rnn['mask'],
            nonlinearity = lasagne.nonlinearities.tanh,
            learn_init=True, grad_clipping=100., backwards=True)

        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        rnn['sum'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward'], rnn['rnn_backward']], operation)

        # First, retrieve symbolic variables for the input shape
        n_batch, n_time_steps, n_features = rnn['input'].input_var.shape
        # Now, squash the n_batch and n_time_steps dimensions
        rnn['reshape'] = lasagne.layers.ReshapeLayer(rnn['sum'], (-1, N_HIDDEN))
        # Now, we can apply feed-forward layers as usual.
        # We want the network to predict a single value, the sum, so we'll use a single unit.
        rnn['dense'] = lasagne.layers.DenseLayer(
            rnn['reshape'], num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
        # Now, the shape will be n_batch*n_timesteps, 1.  We can then reshape to
        # n_batch, n_timesteps to get a single value for each timstep from each sequence
        rnn['out'] = lasagne.layers.ReshapeLayer(rnn['dense'], (n_batch, n_time_steps))

        return rnn

    def fit(self, X, y, NUM_EPOCHS=10, EPOCH_SIZE=100):
        self.rnn = self.make_rnn(X, y)
        train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y, test_size=0.1, random_state=42)
        # lasagne.layers.get_output produces an expression for the output of the net
        network_output = lasagne.layers.get_output(self.rnn['out'])
        # The value we care about is the final value produced for each sequence
        # so we simply slice it out.
        predicted_values = network_output[:, -1]
        # Our cost will be mean-squared error
        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.rnn['out'])
        # Compute adam updates for training
        updates = lasagne.updates.adam(cost, all_params)
        # Theano functions for training and computing cost
        train = theano.function(
            [self.rnn['input'].input_var, target_values, self.rnn['mask'].input_var],
            cost, updates=updates)
        compute_cost = theano.function(
            [self.rnn['input'].input_var, target_values, self.rnn['mask'].input_var],
            cost)
        mask_train = np.ones((train_set_x.shape[0], train_set_x.shape[2]))
        mask_valid = np.ones((valid_set_x.shape[0], valid_set_x.shape[2]))
        print("Beginning training")
        for epoch in range(NUM_EPOCHS):
            for _ in range(EPOCH_SIZE):
                train(train_set_x, train_set_y, mask_train)
            cost_val = compute_cost(valid_set_x, valid_set_y, mask_valid)
            print("Epoch {} validation cost = {}".format(epoch + 1, cost_val))

    def predict(self, X):
        mask_predict = np.ones((X.shape[0], X.shape[2]))
        predict = theano.function(
                inputs=[self.rnn['input'].input_var, self.rnn['mask'].input_var],
                outputs= lasagne.layers.get_output(self.rnn['out']))
        prediction = predict(X, mask_predict)
        return prediction[:, -1]
