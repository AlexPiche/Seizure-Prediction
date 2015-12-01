import theano
import theano.tensor as T
import lasagne
import numpy as np
import sklearn.datasets
import os
import matplotlib.pyplot as plt

# Recurrent Networks

# Our LSTM will have 10 hidden/cell units
N_HIDDEN = 10
N_FEATURES_DIM = X.shape[2]


l_in = lasagne.layers.InputLayer(shape=(None, None, N_FEATURES_DIM))
# This input will be used to provide the network with masks.
# Masks are expected to be matrices of shape (n_batch, n_time_steps);
# both of these dimensions are variable for us so we will use
# an input shape of (None, None)
l_mask = lasagne.layers.InputLayer(shape=(None, None))


# All gates have initializers for the input-to-gate and hidden state-to-gate
# weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
# The convention is that gates use the standard sigmoid nonlinearity,
# which is the default for the Gate class.
print("Building the model")
gate_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    b=lasagne.init.Constant(0.))

cell_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    # Setting W_cell to None denotes that no cell connection will be used.
    W_cell=None, b=lasagne.init.Constant(0.),
    # By convention, the cell nonlinearity is tanh in an LSTM.
    nonlinearity=lasagne.nonlinearities.tanh)


# Bidirectional LSTM

l_lstm = lasagne.layers.recurrent.LSTMLayer(
    l_in, N_HIDDEN,
    # We need to specify a separate input for masks
    mask_input=l_mask,
    # Here, we supply the gate parameters for each gate
    ingate=gate_parameters, forgetgate=gate_parameters,
    cell=cell_parameters, outgate=gate_parameters,
    # We'll learn the initialization and use gradient clipping
    learn_init=True, grad_clipping=100.)

l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
    l_in, N_HIDDEN, ingate=gate_parameters,
    mask_input=l_mask, forgetgate=gate_parameters,
    cell=cell_parameters, outgate=gate_parameters,
    learn_init=True, grad_clipping=100., backwards=True)

# We'll combine the forward and backward layer output by summing.
# Merge layers take in lists of layers to merge as input.
l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

# First, retrieve symbolic variables for the input shape
n_batch, n_time_steps, n_features = l_in.input_var.shape
# Now, squash the n_batch and n_time_steps dimensions
l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))
# Now, we can apply feed-forward layers as usual.
# We want the network to predict a single value, the sum, so we'll use a single unit.
l_dense = lasagne.layers.DenseLayer(
    l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
# Now, the shape will be n_batch*n_timesteps, 1.  We can then reshape to
# n_batch, n_timesteps to get a single value for each timstep from each sequence
l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_time_steps))



# Symbolic variable for the target network output.
# It will be of shape n_batch, because there's only 1 target value per sequence.
target_values = T.vector('target_output')
# This matrix will tell the network the length of each sequences.
# The actual values will be supplied by the gen_data function.
mask = T.matrix('mask')

# lasagne.layers.get_output produces an expression for the output of the net
network_output = lasagne.layers.get_output(l_out)
# The value we care about is the final value produced for each sequence
# so we simply slice it out.
predicted_values = network_output[:, -1]
# Our cost will be mean-squared error
cost = T.mean((predicted_values - target_values)**2)
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
# Compute adam updates for training
updates = lasagne.updates.adam(cost, all_params)
# Theano functions for training and computing cost
train = theano.function(
    [l_in.input_var, target_values, l_mask.input_var],
    cost, updates=updates)
compute_cost = theano.function(
    [l_in.input_var, target_values, l_mask.input_var], cost)

# We'll use this "validation set" to periodically check progress

from load_data import load_subjects, load_subjects_test
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation

from Utilities import make_csv_predictions,train_predict

import numpy as np

print("Getting the data")
all_patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']

""" you can load all the subjects you want by putting them in a list
	the return value is a dictionary with the subjects as keys """
all_subjects_dict = load_subjects(all_patients)

all_subjects_dict_test = load_subjects_test(all_patients)

""" pat dict is a dictionary that has the different preprocessing techniques as its keys """
pat1_dict = all_subjects_dict['Patient_1']

pat1_dict_test = all_subjects_dict_test['Patient_1']

""" the different preprocessing techniques, michael hills created a classifier for each and then did an average """
#print pat1_dict.keys()

"""	the data itself is separated by whether it was interictal or preictal """
correlation_data_interictal, correlation_data_preictal = pat1_dict['corr']

i, p = pat1_dict['fft_mag_fbin-mean']
X = np.vstack((i, p))
ones = np.ones(p.shape[0])
zeros = np.zeros(i.shape[0])

y = np.append(zeros,ones)

mask = np.ones((X.shape[0], X.shape[2]))

""" Loop over all patients, 
	make probabilistic predictions for each method within a given patient
	combine the probalities by either:
	1) just summing them up
	2) subtracting the mean prediction probability from each class 
	   for each method and then summing (Avoids all 0 predictions) """

# We'll train the network with 10 epochs of 100 minibatches each
NUM_EPOCHS = 10
EPOCH_SIZE = 100
print("Beginning training")
for epoch in range(NUM_EPOCHS):
    for _ in range(EPOCH_SIZE):
        train(X, y, mask)
    cost_val = compute_cost(X, y, mask)
    print("Epoch {} validation cost = {}".format(epoch + 1, cost_val))


