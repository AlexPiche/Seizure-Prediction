from load_data import load_subjects, load_subjects_test
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation

from Utilities import make_csv_predictions,train_predict

import numpy as np

all_patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']

""" you can load all the subjects you want by putting them in a list
	the return value is a dictionary with the subjects as keys """
all_subjects_dict = load_subjects(all_patients)

all_subjects_dict_test = load_subjects_test(all_patients)

""" pat dict is a dictionary that has the different preprocessing techniques as its keys """
pat1_dict = all_subjects_dict['Patient_1']

pat1_dict_test = all_subjects_dict_test['Patient_1']

""" the different preprocessing techniques, michael hills created a classifier for each and then did an average """
print pat1_dict.keys()

"""	the data itself is separated by whether it was interictal or preictal """
correlation_data_interictal, correlation_data_preictal = pat1_dict['corr']

test_cor = pat1_dict_test['corr']

""" Loop over all patients, 
	make probabilistic predictions for each method within a given patient
	combine the probalities by either:
	1) just summing them up
	2) subtracting the mean prediction probability from each class 
	   for each method and then summing (Avoids all 0 predictions) """
all_predictions = []
for patient in all_patients:

	patient_data = all_subjects_dict[patient]
	patient_data_test = all_subjects_dict_test[patient]

	predictions = train_predict(patient_data,patient_data_test,LogisticRegression(),
		flatten = True, enhance = True, subtract_mean = True)

	all_predictions.append(predictions)

""" takes in a list of lists of predictions and a list of patients, 
	outputs a csv that can be submitted to kaggle """
make_csv_predictions(all_predictions,all_patients)

	

