from load_data import load_subjects

patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1']
#you can load all the subjects you want by putting them in a list
#the return value is a dictionary with the subjects as keys.
all_subjects_dict = load_subjects(patients)

#dog1 dict is a dictionary that has the different preprocessing techniques as its keys 
dog1_dict = all_subjects_dict['Dog_1']

#the different preprocessing techniques, michael hills created a classifier for each and then did an averag
print dog1_dict.keys()

#the data itself is separated by whether it was interictal or preictal
correlation_data_interictal, correlation_data_preictal = dog1_dict['corr']

