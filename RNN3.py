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

class LSTM:
    def __init__(self):
        self.lstm = None
    def make_lstm(self, X, y):
        print "Building the Model"
        lstm = {}
        N_HIDDEN = 50
        N_FEATURES_DIM = X.shape[2]
        lstm['input'] = lasagne.layers.InputLayer(shape=(None, None, N_FEATURES_DIM))
        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (n_batch, n_time_steps);
        # both of these dimensions are variable for us so we will use
        # an input shape of (None, None)
        lstm['mask'] = lasagne.layers.InputLayer(shape=(None, None))
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
            # By convention, the cell nonlinearity is tanh in an LSTM.
            nonlinearity=lasagne.nonlinearities.tanh)

        # Add noise to the input
        lstm['noise'] = lasagne.layers.GaussianNoiseLayer(lstm['input'], sigma=0.1)

        # Bidirectional LSTM

        lstm['lstm_forward'] = lasagne.layers.recurrent.LSTMLayer(
            lstm['noise'], N_HIDDEN,
            # We need to specify a separate input for masks
            mask_input=lstm['mask'],
            # Here, we supply the gate parameters for each gate
            ingate=gate_parameters, forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            # We'll learn the initialization and use gradient clipping
            learn_init=True, grad_clipping=100.)

        lstm['lstm_backward'] = lasagne.layers.recurrent.LSTMLayer(
            lstm['noise'], N_HIDDEN, ingate=gate_parameters,
            mask_input=lstm['mask'], forgetgate=gate_parameters,
            cell=cell_parameters, outgate=gate_parameters,
            learn_init=True, grad_clipping=100., backwards=True)

        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        lstm['sum'] = lasagne.layers.ElemwiseSumLayer([lstm['lstm_forward'], lstm['lstm_backward']])

        # First, retrieve symbolic variables for the input shape
        n_batch, n_time_steps, n_features = lstm['input'].input_var.shape
        # Now, squash the n_batch and n_time_steps dimensions
        lstm['reshape'] = lasagne.layers.ReshapeLayer(lstm['sum'], (-1, N_HIDDEN))
        # Now, we can apply feed-forward layers as usual.
        # We want the network to predict a single value, the sum, so we'll use a single unit.
        lstm['dense'] = lasagne.layers.DenseLayer(
            lstm['reshape'], num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
        # Now, the shape will be n_batch*n_timesteps, 1.  We can then reshape to
        # n_batch, n_timesteps to get a single value for each timstep from each sequence
        lstm['out'] = lasagne.layers.ReshapeLayer(lstm['dense'], (n_batch, n_time_steps))

        return lstm

    def fit(self, X, y):
        self.lstm = self.make_lstm(X, y)
        print "Training"
        train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y, test_size=0.1, random_state=42)
        # lasagne.layers.get_output produces an expression for the output of the net
        network_output = lasagne.layers.get_output(self.lstm['out'])
        # The value we care about is the final value produced for each sequence
        # so we simply slice it out.
        predicted_values = network_output[:, -1]
        # Our cost will be mean-squared error
        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.lstm['out'])
        # Compute adam updates for training
        updates = lasagne.updates.adam(cost, all_params)
        # Theano functions for training and computing cost
        train = theano.function(
            [self.lstm['input'].input_var, target_values, self.lstm['mask'].input_var],
            cost, updates=updates)
        compute_cost = theano.function(
            [self.lstm['input'].input_var, target_values, self.lstm['mask'].input_var],
            cost)
        NUM_EPOCHS = 100
        EPOCH_SIZE = 1000
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
                inputs=[self.lstm['input'].input_var, self.lstm['mask'].input_var],
                outputs= lasagne.layers.get_output(self.lstm['out']))
        prediction = predict(X, mask_predict)
        return prediction[:, -1]


class GRU:
    def __init__(self):
        self.gru = None
    def make_gru(self, X, y):
        print "Building the Model"
        gru = {}
        N_HIDDEN = 50
        N_FEATURES_DIM = X.shape[2]
        gru['input'] = lasagne.layers.InputLayer(shape=(None, None, N_FEATURES_DIM))
        # This input will be used to provide the network with masks.
        # Masks are expected to be matrices of shape (n_batch, n_time_steps);
        # both of these dimensions are variable for us so we will use
        # an input shape of (None, None)
        gru['mask'] = lasagne.layers.InputLayer(shape=(None, None))
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
            # By convention, the cell nonlinearity is tanh in an GRU.
            nonlinearity=lasagne.nonlinearities.tanh)

        # Add noise to the input
        gru['noise'] = lasagne.layers.GaussianNoiseLayer(gru['input'], sigma=0.1)

        # Bidirectional GRU

        gru['gru_forward'] = lasagne.layers.recurrent.GRULayer(
            gru['noise'], N_HIDDEN, mask_input=gru['mask'],
            resetgate=gate_parameters , updategate=gate_parameters,
            learn_init=True, grad_clipping=100.)

        gru['gru_backward'] = lasagne.layers.recurrent.GRULayer(
            gru['noise'], N_HIDDEN, mask_input=gru['mask'],
            learn_init=True, grad_clipping=100., backwards=True)

        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        gru['sum'] = lasagne.layers.ElemwiseSumLayer([gru['gru_forward'], gru['gru_backward']])

        # First, retrieve symbolic variables for the input shape
        n_batch, n_time_steps, n_features = gru['input'].input_var.shape
        # Now, squash the n_batch and n_time_steps dimensions
        gru['reshape'] = lasagne.layers.ReshapeLayer(gru['sum'], (-1, N_HIDDEN))
        # Now, we can apply feed-forward layers as usual.
        # We want the network to predict a single value, the sum, so we'll use a single unit.
        gru['dense'] = lasagne.layers.DenseLayer(
            gru['reshape'], num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
        # Now, the shape will be n_batch*n_timesteps, 1.  We can then reshape to
        # n_batch, n_timesteps to get a single value for each timstep from each sequence
        gru['out'] = lasagne.layers.ReshapeLayer(gru['dense'], (n_batch, n_time_steps))

        return gru

    def fit(self, X, y):
        self.gru = self.make_gru(X, y)
        print "Training"
        train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y, test_size=0.1, random_state=42)
        # lasagne.layers.get_output produces an expression for the output of the net
        network_output = lasagne.layers.get_output(self.gru['out'])
        # The value we care about is the final value produced for each sequence
        # so we simply slice it out.
        predicted_values = network_output[:, -1]
        # Our cost will be mean-squared error
        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.gru['out'])
        # Compute adam updates for training
        updates = lasagne.updates.adam(cost, all_params)
        # Theano functions for training and computing cost
        train = theano.function(
            [self.gru['input'].input_var, target_values, self.gru['mask'].input_var],
            cost, updates=updates)
        compute_cost = theano.function(
            [self.gru['input'].input_var, target_values, self.gru['mask'].input_var],
            cost)
        NUM_EPOCHS = 100
        EPOCH_SIZE = 1000
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
                inputs=[self.gru['input'].input_var, self.gru['mask'].input_var],
                outputs= lasagne.layers.get_output(self.gru['out']))
        prediction = predict(X, mask_predict)
        return prediction[:, -1]
