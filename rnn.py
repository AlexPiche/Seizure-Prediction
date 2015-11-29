# Based on https://github.com/craffel/Lasagne-tutorial
from __future__ import print_function
import pdb
from load_data import load_subjects
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import cross_validation
import numpy as np
patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1']

#you can load all the subjects you want by putting them in a list
#the return value is a dictionary with the subjects as keys.
all_subjects_dict = load_subjects(patients)

#pat dict is a dictionary that has the different preprocessing techniques as its keys 
pat1_dict = all_subjects_dict['Patient_1']

#the different preprocessing techniques, michael hills created a classifier for each and then did an averag

#the data itself is separated by whether it was interictal or preictal
correlation_data_interictal, correlation_data_preictal = pat1_dict['corr']

#classifiers = [LinearRegression(),LogisticRegression()]


#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''



import numpy as np
import theano
import theano.tensor as T
import lasagne


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 1280
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 504
n_batch = 100

# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10


def main(num_epochs=NUM_EPOCHS):
        print("Building network ...")
        # First, we build the network, starting with an input layer
        # Recurrent layers expect input of shape
        # (batch size, max sequence length, number of features)
        l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
        # The network also needs a way to provide a mask for each sequence.  We'll
        # use a separate input layer for that.  Since the mask only determines
        # which indices are part of the sequence for each batch entry, they are
        # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
        l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
        # We're using a bidirectional network, which means we will combine two
        # RecurrentLayers, one with the backwards=True keyword argument.
        # Setting a value for grad_clipping will clip the gradients in the layer
        l_forward = lasagne.layers.RecurrentLayer(
            l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh)
        l_backward = lasagne.layers.RecurrentLayer(
            l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
        # The objective of this task depends only on the final value produced by
        # the network.  So, we'll use SliceLayers to extract the LSTM layer's
        # output after processing the entire input sequence.  For the forward
        # layer, this corresponds to the last value of the second (sequence length)
        # dimension.
        l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)
        # For the backwards layer, the first index actually corresponds to the
        # final output of the network, as it processes the sequence backwards.
        l_backward_slice = lasagne.layers.SliceLayer(l_backward, 0, 1)
        # Now, we'll concatenate the outputs to combine them.
        l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice])
        # Our output layer is a simple dense connection, with 1 output unit
        l_out = lasagne.layers.DenseLayer(
            l_sum, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

        target_values = T.vector('target_output')

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(l_out)
        # The value we care about is the final value produced for each sequence
        predicted_values = network_output[:, -1]
        # Our cost will be mean-squared error
        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(l_out)
        # Compute SGD updates for training
        print("Computing updates ...")
        updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
        compute_cost = theano.function(
            [l_in.input_var, target_values, l_mask.input_var], cost)

        # We'll use this "validation set" to periodically check progress

        print("Training ...")
        try:
                for epoch in range(num_epochs):
                        for _ in range(EPOCH_SIZE):
                                for patient in patients:

                                        cur_patient_data = all_subjects_dict[patient]

                                        for method in cur_patient_data:
            
                                                i,p = cur_patient_data[method]
                                                shape_i = i.shape
                                                shape_p = p.shape

                                                #change the shape from (#of examples,#channels,#time) to (#examples,channels*time)
                                                interictal = np.reshape(i,(shape_i[0],np.product(shape_i[1:])))
                                                preictal = np.reshape(p,(shape_p[0],np.product(shape_p[1:])))

                                                X = np.vstack((interictal,preictal))

                                                ones = np.ones(preictal.shape[0])
                                                zeros = np.zeros(interictal.shape[0])

                                                y = np.append(zeros,ones)
                                                mask = np.ones(X.shape)

                                                X.astype(theano.config.floatX)
                                                y.astype(theano.config.floatX)
                                                mask.astype(theano.config.floatX)
                                                pdb.set_trace()

                                                train(X, y, mask)
                                                cost_val = compute_cost(X, y, mask)
                                                print("Epoch {} validation cost = {}".format(epoch, cost_val))
        except KeyboardInterrupt:
                pass

if __name__ == '__main__':
        main()
