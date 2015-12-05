from load_data import load_subjects
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation
import os

from Utilities_RNN import make_csv_predictions,train_predict_test,find_k_best_methods

import numpy as np

from CNN import CNN
from RNN import RNN

print("Getting the data")
all_patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']

""" you can load all the subjects you want by putting them in a list
	the return value is a dictionary with the subjects as keys """
all_subjects_dict = load_subjects(all_patients)
#all_subjects_dict_cnn = load_subjects(all_patients,cnn =True)

all_subjects_dict_test = load_subjects(all_patients,test=True)
#all_subjects_dict_test_cnn = load_subjects(all_patients,test=True,cnn=True)

""" pat dict is a dictionary that has the different preprocessing techniques as its keys """
pat1_dict = all_subjects_dict['Patient_1']

pat1_dict_test = all_subjects_dict_test['Patient_1']

""" the different preprocessing techniques, michael hills created a classifier for each and then did an average """
#print pat1_dict.keys()

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
best_methods = []
best_scores = []
models = {'lstm1':['lstm',10,150], 'lstm2':['lstm',50,150],
          'lstm3':['lstm',100,150],'lstm4':['lstm',150,150],'gru1':['gru',10,150],
          'gru2':['gru',50,150],'gru3':['gru',100,150], 'gru4':['gru',150,150],
          'vanilla1':['vanilla',10,150],'vanilla2':['vanilla',50,150],'vanilla3':['vanilla',100,150], 'vanilla4':['vanilla',150,150]}

for model in models:
        for patient in all_patients:
                global file_title
                file_title = 'analysis/'+str(models[model][0]) + str(models[model][1]) + str(patient) + '.csv'

                try:
                        os.remove(file_title)
                except OSError:
                        pass

                open(file_title, 'a').close()
                with open(file_title, "a") as myfile:
                        myfile.write('epoch')
                        for epoch in range(models[model][2]):
                                myfile.write(','+str(epoch+1) )


                patient_data = all_subjects_dict[patient]
                patient_data_test = all_subjects_dict_test[patient]

                "set best methods to be the number of preprocessing methods you want to use."
                methods,scores,predictions = train_predict_test(patient_data,patient_data_test, RNN(flavour=models[model][0], N_HIDDEN=models[model][1], NUM_EPOCHS=models[model][2],patient=patient),
                                                                flatten = False, enhance_size = 500, subtract_mean = False,best_methods=0,probability=False,cnn=False)

                best_methods.extend(methods)
                best_scores.extend(scores)
                all_predictions.append(predictions)

                """ takes in a list of lists of predictions and a list of patients, 
                outputs a csv that can be submitted to kaggle """
                make_csv_predictions(all_predictions,all_patients,models[model][0])
                for bm,bs in zip(best_methods,best_scores):
                        print bm,bs

