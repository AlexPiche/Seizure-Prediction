import theano
import os
from sklearn.cross_validation import train_test_split
import theano.tensor as T
import lasagne
import numpy as np

target_values = T.vector('target_output')
mask = T.matrix('mask')

# Recurrent Networks

class RNN:
    def __init__(self, flavour='lstm', N_HIDDEN=50, NUM_EPOCHS=10, patient=None):
        self.rnn = None
        self.__N_HIDDEN = N_HIDDEN
        self.__flavour = flavour
        self.__NUM_EPOCHS = NUM_EPOCHS
        global file_title
        file_title = "analysis/" + str(flavour) + str(N_HIDDEN) + str(patient) + '.csv'
    def make_rnn(self, X, y, operation = T.add, peepholes=True):
        """
        Build a RNN

        flavour can be 'lstm', 'gru', or vanilla
        operation can be T.add, T.mul, T.minimum, or T.maximum
        """
        N_HIDDEN = self.__N_HIDDEN
        flavour = self.__flavour


        print "Building the Model"
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

        rnn['noise'] = lasagne.layers.GaussianNoiseLayer(rnn['input'], sigma=0.1)

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

        rnn['sum'] = lasagne.layers.ElemwiseMergeLayer([rnn['rnn_forward'], rnn['rnn_backward']], operation)

        n_batch, n_time_steps, n_features = rnn['input'].input_var.shape
        rnn['reshape'] = lasagne.layers.ReshapeLayer(rnn['sum'], (-1, N_HIDDEN))
        rnn['dense'] = lasagne.layers.DenseLayer(
            rnn['reshape'], num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

        # TODO ain't it suppose to be softmax for the last layer??
        rnn['out'] = lasagne.layers.ReshapeLayer(rnn['dense'], (n_batch, n_time_steps))

        return rnn

    def fit(self, X, y, key, EPOCH_SIZE=25, max_patience = 5):
        with open(file_title, "a") as myfile:
                    myfile.write('\n')
                    myfile.write(str(key))
        NUM_EPOCHS=self.__NUM_EPOCHS
        self.rnn = self.make_rnn(X, y)
        # initiate best loss value to 0 correct prediction
        best_val = 1000
        #initialize patience to 0
        patience = 0
        train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y, test_size=0.1, random_state=42)
        network_output = lasagne.layers.get_output(self.rnn['out'])
        predicted_values = network_output[:, -1]
        #cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
        cost = T.mean((predicted_values - target_values)**2)
        all_params = lasagne.layers.get_all_params(self.rnn['out'])
        updates = lasagne.updates.adam(cost, all_params)
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
            with open(file_title, "a") as myfile:
                    myfile.write(','+str(cost_val))
            print("Epoch {} validation cost = {}".format(epoch + 1, cost_val))
            if cost_val < best_val:
                patience = 0
                best_val = cost_val
                best_model = lasagne.layers.get_all_param_values(self.rnn['out'])
            else:
                patience += 1
            if patience > max_patience:
                lasagne.layers.set_all_param_values(self.rnn['out'], best_model)
                break

    def predict_proba(self, X):
        mask_predict = np.ones((X.shape[0], X.shape[2]))
        predict = theano.function(
                inputs=[self.rnn['input'].input_var, self.rnn['mask'].input_var],
                outputs= lasagne.layers.get_output(self.rnn['out']))
        prediction = predict(X, mask_predict)
        print prediction[:, -1]
        return prediction[:, -1]

