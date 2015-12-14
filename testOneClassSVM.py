from load_data import load_subjects
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from utils.loader import load_grouped_train_data, load_train_data, load_test_data
from UtilitiesCNN import make_csv_predictions, train_predict_test_cnn,train_predict_test
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from CNN import CNN
from sklearn.svm import OneClassSVM


all_patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']

""" Loop over all patients, 
	make probabilistic predictions for each method within a given patient
	combine the probalities by either:
	1) just summing them up
	2) subtracting the mean prediction probability from each class 
	   for each method and then summing (Avoids all 0 predictions) """
all_predictions = []
all_predictions_ns = []
validations_true = []
validations_preds = []

for patient in all_patients:

	#LOAD DATA
	d = load_train_data('preprocessed/cnn/', patient)
	x, y, filename_to_idx = d['x'], d['y'], d['filename_to_idx']
	x_test = load_test_data('preprocessed/cnn/', patient)['x']


	test_preds_ns,val_preds, val_true= train_predict_test(
		patient,OneClassSVM(),x,x_test,enhance_size = 0)

	
	all_predictions_ns.append(test_preds_ns)
	validations_preds.append(val_preds)
	validations_true.append(val_true)
	
""" takes in a list of lists of predictions and a list of patients, 
	outputs a csv that can be submitted to kaggle """

make_csv_predictions(all_predictions_ns,all_patients)


