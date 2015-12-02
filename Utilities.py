import numpy as np
import os
import gzip
from sklearn.metrics import accuracy_score
from cross_validation import cross_val_apply,cross_val_predict
from scipy import stats
def enhance_data(data,reference_size,cnn=False):
	"""  add gaussian noise """
	data_temp = data
	i = 0
	while i < reference_size:
		rand = np.random.randint(0,data_temp.shape[0])
		example = data[rand]
		noise = np.random.normal(0,.1,example.shape)
		new_example = example +noise
		if cnn:
			new_example = np.reshape(new_example,(1,example.shape[0],example.shape[1],example.shape[2]))
		else:
			new_example = np.reshape(new_example,(1,example.shape[0],example.shape[1]))

		data = np.append(data,new_example,axis=0)
		i+=1
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
	diff0 = 50 - np.median(class0)
	diff1 = 50 - np.median(class1)
	array[:,0] = class0 + diff0
	array[:,1] = class1 + diff1
	return array

def subtract_mean_probs(array):
	class0 = array[:,0]
	class1 = array[:,1]
	array[:,0] = class0 - np.mean(class0)
	array[:,1] = class1 - np.mean(class1)

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
	if enhance_size > 0:
		p = enhance_data(p,enhance_size,cnn=cnn)
		i = enhance_data(i,enhance_size,cnn=cnn)
	X = np.vstack((i,p))

	if flatten:
		X = flatten_data(X)
		if t is not None:
			t = flatten_data(t)
	ones = np.ones(p.shape[0])
	zeros = np.zeros(i.shape[0])
	y = np.append(zeros,ones)

	return X,y,t

def voting_combination(list_preds):
	
	num_predictors = len(list_preds)
	summed_array = np.zeros(len(list_preds[0]))
	for pred in list_preds:
		summed_array += pred

	return np.array([1 if x > np.floor(num_predictors/2) else 0 for x in summed_array])

def cross_val_list(patient_train,clf,flatten ,enhance_size,subtract_mean, folds, probability):
	""" uses cross validation on all methods
		returns list of tuples of (method,score) """
	
	methods_scores_preds = []
	y = []
	for method in patient_train:
		
		i,p = patient_train[method]

		X,y,_ = data_process(i,p,enhance_size,flatten)

		
		if probability:
			preds_proba = cross_val_apply(clf,X,y,apply_func='predict_proba',cv = folds)
			if subtract_mean:
				preds_proba = subtract_mean_probs(preds_proba)
			preds = max_prob_over_classes(preds_proba)
			score = accuracy_score(y,preds)
			methods_scores_preds.append((method,score,preds))
		else:
			preds = cross_val_predict(clf,X,y,cv=folds)
			score = accuracy_score(y,preds)
			methods_scores_preds.append((method,score,preds))
			
		
	return methods_scores_preds,y

def find_k_best_methods(k,patient_train,clf,flatten ,enhance_size,subtract_mean,folds, probability,combined = False):
	
	methods_scores_preds,y = cross_val_list(patient_train,clf,flatten,enhance_size,subtract_mean,folds,probability)
	methods_scores_preds.sort(key=lambda tup: tup[1])	

	best_method_scores_preds = methods_scores_preds[:k]

	list_preds = [x[2] for x in best_method_scores_preds]
	
	best_methods = [x[0] for x in best_method_scores_preds]
	best_scores = [x[1] for x in best_method_scores_preds]
	
	combined_preds = voting_combination(list_preds)
	# print 'combined' + str(k) +'best' + str(score)
	if combined:
		score = accuracy_score(y,combined_preds)
		best_methods.append('combined')
		best_scores.append(score)
	
	return best_methods,best_scores,combined_preds
	
	


def train_predict_test(patient_train,patient_test,clf,
	flatten = True,enhance_size = 0,subtract_mean = False,
	best_methods = 0, probability = True,folds =2, cnn=False):

	""" loop over all preprocessing methods for a given patient
		enhance data by given size
		flatten the data if your classifier takes 2D data (samples,features)
		add the probabilities from each pre processing method
		return the maximum probability for each test example """

	preds_all_methods = []
	best_method_scores = []

	best_methods_list = []
	best_scores = []
	patient_keys = patient_train.keys()

	if best_methods > 0:
		best_methods_list,best_scores,combined_preds = find_k_best_methods(best_methods,patient_train,clf,
			flatten,enhance_size,subtract_mean, folds,probability)
		patient_keys = best_methods_list

	for key in patient_keys:
		i,p = patient_train[key]
		test = patient_test[key]
		print i.shape,p.shape
		X,y,t = data_process(i,p,enhance_size=enhance_size,flatten=flatten,t=test,cnn=cnn)
		clf.fit(X,y)
		if probability:
			preds_proba = clf.predict_proba(t)
			if subtract_mean:
				preds_proba = subtract_mean_probs(preds_proba)
			preds = max_prob_over_classes(preds_proba)
			preds_all_methods.append(preds)
		else:
			preds = clf.predict(t)
			preds_all_methods.append(preds)
	
	test_preds = voting_combination(preds_all_methods)


	return best_methods_list,best_scores,test_preds


	

		
	
	

