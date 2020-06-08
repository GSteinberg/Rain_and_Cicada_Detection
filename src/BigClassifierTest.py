# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:06:46 2018

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
trainSet="/Path/To/Trainset/TrainSet.arff"
testSet="/Path/To/TestSet/TestSet.arff"
textFile=open(trainSet,"r")
line=textFile.readline()
line=textFile.readline()
attIDs=[]
lineNum=1
while "@data" not in line:
    line=textFile.readline()
    if "yes" not in line and "@data" not in line:
        if "mfcc" in line:      #Insert relevant attribute rule here for dataset
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
scaler=StandardScaler()
attributes=scaler.fit_transform(attributes)
dump(scaler,"CicScale.file")
pca=PCA(n_components=0.9,svd_solver="full")         #Insert pre-procssing steps here
attributes=pca.fit_transform(attributes)
#dump(pca,"PCA.file")
#nn=neighbors.KNeighborsClassifier(n_neighbors=10,weights="uniform",p=3)
nn=svm.SVC(C=1000,kernel='poly',degree=5,gamma=0.01,probability=True)   #Insert relevant classifier here
nn=nn.fit(attributes,classes)
print(nn.score)
dump(nn,"NNCic.file")
testFile=open(testSet,'r')
line=testFile.readline()
line=testFile.readline()
attIDs=[]
attList=[]
names=[]
lineNum=1
while "@data" not in line:
    line=testFile.readline()
    if "yes" not in line and "@data" not in line:
        if "mfcc" in line:      #Insert relevant attribute rule here for dataset
            attIDs.append(lineNum)
    lineNum+=1
line=testFile.readline()
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
    attList.append(tempAtts)
    line=testFile.readline()
newAtts=scaler.transform(attList)
newAtts=pca.transform(newAtts)          #Pre-Processing calculated above from training set
predProb=nn.predict_proba(newAtts)
predictCSV=open("Predictions.csv","w")
for i in range(0,len(names)):
    predictCSV.write(str(names[i])+","+str(round(predProb[i][1],4))+"\n")
predictCSV.close()
print("File executed in "+str(time.time()-start))