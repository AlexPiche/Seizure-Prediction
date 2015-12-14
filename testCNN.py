from load_data import load_subjects
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from utils.loader import load_grouped_train_data, load_train_data, load_test_data
from UtilitiesCNN import make_csv_predictions, train_predict_test_cnn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from CNN import CNN

all_patients = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0','1'], rotation=45)
    plt.yticks(tick_marks, ['0','1'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_train_val_loss(train_loss,valid_loss,subject):
	
	plt.title('CNN '+ subject+' train_val_loss')
	
	plt.plot(train_loss, linewidth=3, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.ylim(1e-1, 1e0)
	plt.yscale("log")
	
	

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

	test_preds,test_preds_ns,val_preds, val_true,train_loss,valid_loss= train_predict_test_cnn(
		patient,CNN(patient),x,x_test,enhance_size = 1000)

	roc_area = roc_auc_score(val_true,val_preds)
	print patient, roc_area

	plot = plt.figure()
	plot_train_val_loss(train_loss,valid_loss,patient)
	plot.savefig('./figs/CNN'+patient+'train_val.png')

	
	all_predictions.append(test_preds)
	all_predictions_ns.append(test_preds_ns)
	validations_preds.append(val_preds)
	validations_true.append(val_true)
	
""" takes in a list of lists of predictions and a list of patients, 
	outputs a csv that can be submitted to kaggle """
make_csv_predictions(all_predictions,all_patients)
make_csv_predictions(all_predictions_ns,all_patients)


validations_true_flat = [item for sublist in validations_true for item in sublist]
validations_preds_flat = [item for sublist in validations_preds for item in sublist]

#ROC CURVE
plot = plt.figure()
for patient,val_true,val_preds in zip(all_patients,validations_true,validations_preds):
	roc_area = roc_auc_score(val_true,val_preds)
	fpr,tpr,threshold = roc_curve(val_true,val_preds)
	print patient + "ROC_AUC", roc_area 
	plt.plot(fpr, tpr, lw=1, label='ROC' +patient +'(area = %0.2f)' % (roc_area))

roc_area = roc_auc_score(validations_true_flat,validations_preds_flat)
fpr,tpr,threshold = roc_curve(validations_true_flat,validations_preds_flat)
print "All Patients ROC_AUC", roc_area 
plt.plot(fpr, tpr, lw=1, label='ROC All Patients' +'(area = %0.2f)' % (roc_area))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CNN')
plt.legend(loc="lower right")
plot.savefig('./figs/ROC.png')

