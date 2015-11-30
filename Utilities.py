import numpy as np
import os
import gzip


def enhance_data(data,reference_size):
	""" right now just appends the preictal set over and over again untill its ~ even with interictal
		will add gaussian noise later"""
	data_temp = data
	while data.shape[0] < reference_size:
		data = np.vstack((data,data_temp))
	return data


def flatten_data(X):
	""" change the shape from (#of examples,#channels,#time) to (#examples,channels*time)"""

	shape_x = X.shape
	X = np.reshape(X,(shape_x[0],np.product(shape_x[1:])))
	return X


def make_csv_for_target_predictions(target, predictions):
	""" formats the prediction into the required string format for a given target (ie. Dog_1)"""

	return ['%s_test_segment_%.4d.mat,%.10f' % (target, i+1, p) for i, p in enumerate(predictions)]

def make_csv_predictions(all_predictions,all_patients):
	""" takes in predictions as list or array"""

	all_predictions_string = ['clip,preictal']

	for patient,predictions in zip(all_patients,all_predictions):
		all_predictions_string.append('\n'.join(make_csv_for_target_predictions(patient,predictions)))
	id = 0
	done = False
	while not done:
		try:
			filename = 'submission.csv.gz'+str(id)
			guesses = '\n'.join(all_predictions_string)
			fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0644)
			os.close(fd)

			f = gzip.open(filename, 'wb')
			f.write(guesses)
			f.close()
			done = True

		except OSError, e:
			id += 1


def max_prob_over_classes(probs_array):
	""" takes the index that corresponds to the maximum value over each row """	
	
	max_list = [row.argmax(axis=0) for row in probs_array]

	return max_list


def sum_probabilities(prob_list_arrays,subtract_mean = False):
	""" given a list of probability array (examples,classes)
		add them up, unless subtract mean is enabled
		in which case subtract the mean class probability for each preprocessing method first
		(simple way to avoid predicting all zeros, im not sure if this makes sense)"""

	predictions = []

	probs_array = np.zeros(prob_list_arrays[0].shape)

	for array in prob_list_arrays:

		if subtract_mean:
			class0 = array[:,0]
			class1 = array[:,1]
			array[:,0] = class0 - np.mean(class0)
			array[:,1] = class1 - np.mean(class1)
		
		probs_array += array

	return probs_array

def train_predict(patient_train,patient_test,clf,flatten = True,enhance = False,subtract_mean = False):
	""" loop over all preprocessing methods for a given patient
		enhance preictal data if flag is set
		flatten the data if your classifier takes 2D data (samples,features)
		add the probabilities from each pre processing method
		return the maximum probability for each test example """

	preds_probs_all_methods = []
	for method in patient_train:
		
		i,p = patient_train[method]
		
		if enhance_data:
			p = enhance_data(p,i.shape[0])
		
		X = np.vstack((i,p))
		t = patient_test[method]

		if flatten:
			t = flatten_data(t)
			X = flatten_data(X)

		ones = np.ones(p.shape[0])
		zeros = np.zeros(i.shape[0])

		y = np.append(zeros,ones)

		clf.fit(X,y)

		preds_probs_all_methods.append(clf.predict_proba(t))

	combined_methods_probs = sum_probabilities(preds_probs_all_methods,subtract_mean = subtract_mean)
	predictions            = max_prob_over_classes(combined_methods_probs) 
	
	return predictions

