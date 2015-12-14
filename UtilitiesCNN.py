import numpy as np
import os
import gzip
from sklearn.metrics import accuracy_score
from cross_validation import cross_val_apply,cross_val_predict,train_test_split
from scipy import stats
import cPickle
from utils.loader import load_grouped_train_data, load_train_data, load_test_data
from utils.config_name_creator import *
from utils.data_scaler import scale_across_time, scale_across_features
from utils.data_splitter import split_train_valid_filenames, generate_overlapped_data

from sklearn.base import clone

from CNN import CNN
def enhance_data(data_x,data_y,reference_size,cnn=False,even = False):
	"""  add gaussian noise """
	data_temp = data_x
	#print "var", np.var(data_x)
	i = 0
	new_data_x = []
	new_data_y = []
	preictal_indices = data_y == 1
	interictal_indices = data_y ==0
	data_p = data_temp[preictal_indices]
	data_i  = data_temp[interictal_indices]
	var_p = np.var(data_p)
	var_i = np.var(data_i)
	#print data_p.shape
	while i < reference_size:
		rand = np.random.randint(0,data_x.shape[0])
		example = data_x[rand]
		example_y = data_y[rand]
		var = var_i
		if example_y == 1:
			var = var_p
		if even:
			rand = np.random.randint(0,data_p.shape[0])
			example = data_p[rand]
			example_y = 1
			var = var_p
			if i % 2 == 1:
				rand = np.random.randint(0,data_i.shape[0])
				example = data_i[rand]
				example_y = 0
				var = var_i

		noise = np.random.normal(0,1,example.shape)
		new_example = example +noise
		if cnn:
			new_example = np.reshape(new_example,(example.shape[0],example.shape[1],example.shape[2]))
		else:
			new_example = np.reshape(new_example,(example.shape[0],example.shape[1]))

		new_data_x.append(new_example)
		new_data_y.append(example_y)
		i+=1
	X,y = np.array(new_data_x),np.array(new_data_y)
	#print X.shape
	return X,y 


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
			filename = 'submission'+str(id)+'.csv.gz'
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

def set_median_to_half(array):
	class0 = array[:,0]
	class1 = array[:,1]
	diff0 = .5 - np.median(class0)
	diff1 = .5 - np.median(class1)
	array[:,0] = class0 + diff0
	array[:,1] = class1 + diff1
	
	above_1_indices = array > 1
	below_0_indices = array < 0

	array[above_1_indices] = 1
	array[below_0_indices] = 0

	return array

def subtract_mean_probs(array):
	class0 = array[:,0]
	class1 = array[:,1]
	array[:,0] = class0 - np.mean(class0)
	array[:,1] = class1 - np.mean(class1)

	return array

def min_max_scale(array):
	min_p = np.min(array)
	max_p = np.max(array)
	array = (array - min_p) / (max_p - min_p)
	return array
def sum_probabilities(prob_list_arrays,subtract_mean = False):
	""" given a list of probability array (examples,classes)
		add them up, unless subtract mean is enabled
		in which case subtract the mean class probability for each preprocessing method first
		(simple way to avoid predicting all zeros, im not sure if this makes sense)"""

	predictions = []

	probs_array = np.zeros(prob_list_arrays[0].shape)

	for array in prob_list_arrays:

		if subtract_mean:
			array = subtract_mean_probs(array)
		
		probs_array += array

	return probs_array

def data_process(i,p,enhance_size=0,flatten=True,t=None,cnn=False):

	X = np.vstack((i,p))

	ones = np.ones(p.shape[0])
	zeros = np.zeros(i.shape[0])
	y = np.append(zeros,ones)

	if enhance_size > 0:
		X,y = enhance_data(X,y,enhance_size,cnn=cnn)
	
	if flatten:
		X = flatten_data(X)
		if t is not None:
			t = flatten_data(t)
	

	return X,y,t

def voting_combination(list_preds):
	
	num_predictors = len(list_preds)
	summed_array = np.zeros(len(list_preds[0]))
	for pred in list_preds:
		summed_array += pred

	return np.array([1 if x > np.floor(num_predictors/2) else 0 for x in summed_array])

def split_evenly(X,y,test_size = .25):
	preictal_indices = y == 1
	interictal_indices = y ==0
	X_p = X[preictal_indices]
	X_i  = X[interictal_indices]
	y_p = y[preictal_indices]
	y_i = y[interictal_indices]

	num_p = X_p.shape[0] * .25
	test_size_i =  num_p / X_i.shape[0]
	X_p_train, X_p_test,y_p_train,y_p_test = train_test_split(X_p,y_p,test_size=.25,random_state = 33)
	X_i_train, X_i_test,y_i_train,y_i_test = train_test_split(X_i,y_i,test_size=test_size_i,random_state = 39)

	X = np.vstack((X_p_train,X_i_train))
	Xt = np.vstack((X_p_test,X_i_test))

	y = np.append(y_p_train,y_i_train)
	yt = np.append(y_p_test,y_i_test)

	return X,Xt,y,yt


def train_predict_test_cnn(subject,clf,X,X_test,enhance_size = 0):

	filenames_grouped_by_hour = cPickle.load(open('filenames.pickle'))
	data_grouped_by_hour = load_grouped_train_data('preprocessed/cnn/', subject, filenames_grouped_by_hour)

	
	X, y = generate_overlapped_data(data_grouped_by_hour, overlap_size=10,
	                                window_size=X.shape[-1],
	                                overlap_interictal=True,
	                                overlap_preictal=True)

	X, scalers = scale_across_time(X, x_test=None)

	X_test, _ = scale_across_time(X_test, x_test=None, scalers=scalers)


	X,xt,y,yt = split_evenly(X,y,test_size = .25)
	#X,xt,y,yt = train_test_split(X,y,test_size = .25)		
	if enhance_size > 0:
		X,y = enhance_data(X,y,enhance_size,cnn=True,even=True)
		xt,yt = enhance_data(xt,yt,enhance_size,cnn=True,even=True)

	print "train size", X.shape
	print "test_size", xt.shape

	#print "done loading"
	clf.fit(X,y,xt,yt)

	#train_loss = np.array([])
	#valid_loss = np.array([])
	

	#print "train,valid size",train_loss.shape,valid_loss.shape
	#print "done fitting"
	preds_proba = clf.predict_proba(X_test)[:,1]

	# unsup_size = int(X_test.shape[0]/5)
	# top_ind = np.argpartition(preds_proba,-unsup_size)[-unsup_size:]
	# bot_ind = preds_proba.argsort()[:unsup_size]
	# x_new_p = X_test[top_ind]
	# x_new_i = X_test[bot_ind]
	# y_p = np.ones(x_new_p.shape[0])
	# y_i = np.zeros(x_new_i.shape[0])

	#print y_p.shape,y_i.shape
	#print x_new_p.shape, x_new_i.shape
	# x_new = np.vstack((x_new_p,x_new_i))
	# y_new = np.append(y_p,y_i)
	# #print x_new.shape,y_new.shape
	# #X,xt,y,yt = split_evenly(x_new,y_new,test_size = .25)	
	# if enhance_size > 0:
	# 	x_new,y_new = enhance_data(x_new,y_new,enhance_size,cnn=True)
		

	# print "train size", X.shape
	# print "test_size", xt.shape

	# #print "done loading"
	# clf2 = CNN(subject)
	# clf2.fit(x_new,y_new,xt,yt)

	# preds_proba = clf2.predict_proba(X_test)[:,1]
	train_loss = np.array([i["train_loss"] for i in clf.convnet.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in clf.convnet.train_history_])
	#preds_proba = set_median_to_half(preds_proba)[:,1]
	preds_scaled = min_max_scale(preds_proba)
	#print preds_proba.shape
	validation_preds = min_max_scale(clf.predict_proba(xt)[:,1])

	return preds_scaled,preds_proba,list(validation_preds),list(yt),train_loss,valid_loss




	

		
	
	

