# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:06:10 2018

@author: asbrown
"""
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import csv
import sys
import time
from joblib import dump, load
from sklearn import svm, datasets,tree,neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectKBest,f_classif,SelectPercentile,mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB,GaussianNB
from sklearn.decomposition import KernelPCA,PCA
from sklearn.pipeline import Pipeline

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
start=time.time()
trainSet="/Path/To/TrainingData/TrainingData.arff"
testSet="/Path/To/TrainingData/TestingData.arff"
textFile=open(trainSet,"r")
line=textFile.readline()
line=textFile.readline()
attIDs=[]
lineNum=1
while "@data" not in line:
    line=textFile.readline()
    if "yes" not in line and "@data" not in line:
        if "mfcc" in line and "delt" not in line:	#Insert feature rule here
            attIDs.append(lineNum)
    lineNum+=1
    #print(line+" In loop")
    #time.sleep(0.1)
print(line)
names=[]
attributes=[]
classes=[]
line=textFile.readline()
while line:
    #print(line)
    splitLine=line.split(",")
    #print(splitLine[-1])
    #print(splitLine[len(splitLine)-1])
    names.append(splitLine[0])
    tempAtts=[]
    isCicada=0
    for k in range(0,len(splitLine)):
        if k in attIDs:
            tempAtts.append(float(splitLine[k]))
        '''
        if splitLine[i]=="no":
            print(splitLine[i])
        elif splitLine[i]=="yes":
            print(splitLine[i])
        '''
        if "yes" in splitLine[k]:
            isCicada=1
    attributes.append(tempAtts)
    classes.append(isCicada)
    line=textFile.readline()
X = np.asarray(attributes)
y = np.asarray(classes)
scaler=StandardScaler()
attributes=scaler.fit_transform(attributes)
pca=PCA(n_components=0.9)					#Insert pre-processing here
attributes=pca.fit_transform(attributes)
classifier=svm.SVC(C=10,kernel='poly',degree=5,gamma=0.001,probability=True,max_iter=500000,random_state=0)	#Insert classifier here
cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X,y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
rocData=open("ROC2.csv","w")
for i in range(0,np.size(mean_tpr)):
    rocData.write(str(mean_tpr[i])+","+str(mean_fpr[i])+"\n")
rocData.close()
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
print(mean_auc)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()