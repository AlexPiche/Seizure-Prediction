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
print pat1_dict.keys()

#the data itself is separated by whether it was interictal or preictal
correlation_data_interictal, correlation_data_preictal = pat1_dict['corr']

#classifiers = [LinearRegression(),LogisticRegression()]

#for classifier in classifiers:
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

		scores = cross_validation.cross_val_score(LogisticRegression(),X,y,cv=5)

		print patient,"log_reg",scores.mean()

