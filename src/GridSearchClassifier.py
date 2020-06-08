import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import csv
import sys
import time
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

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
locations=["Dataset.arff","DatasetWithHPF.arff","DatasetWithMMSE.arff","DatasetWithHPFMMSE.arff"]
attRules=["All","AllNoDelta","Indices","FreqIndices","mfccs","mfccsNoDelta"]
for i in locations:
    for j in attRules:
        print("LOCATION: "+str(i)+" ATTRULE: "+str(j))
        location="/Path/To/Folder/"+str(i)
        textFile=open(location,"r")
        line=textFile.readline()
        line=textFile.readline()
        attIDs=[]
        lineNum=1
        while "@data" not in line:
            line=textFile.readline()
            if "yes" not in line and "@data" not in line:
                if "Indices" in j:
                    if "mfcc" not in line and ("Freq" in j or not hasNumbers(line)):
                        #print("adding "+str(line))
                        attIDs.append(lineNum)
                elif "mfccs" in j:
                    if "mfcc" in line and ("Delta" in j or "delt" not in line):
                        #print("adding "+str(line))
                        attIDs.append(lineNum)
                elif "delt" not in line or "Delta" not in j:
                        #print("adding "+str(line))
                        attIDs.append(lineNum)
            lineNum+=1
        print(line)
        names=[]
        attributes=[]
        classes=[]
        line=textFile.readline()
        while line:
            splitLine=line.split(",")
            names.append(splitLine[0])
            tempAtts=[]
            isCicada=0
            #print(attIDs)
            for k in range(0,len(splitLine)):
                #print(k)
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
        preProcParams=[PCA(n_components=0.99,svd_solver="full"),PCA(n_components=0.9,svd_solver="full"),None,SelectPercentile(mutual_info_classif,percentile=15),SelectPercentile(mutual_info_classif,percentile=30)]
        aParam_grid=[
                {
                 'preProc':preProcParams,
                 'classify':[neighbors.KNeighborsClassifier()],
                 'classify__n_neighbors':[1,2,3,5,10,50,100,200,400],
                 'classify__weights':["uniform","distance"],
                 'classify__p':[1,2,3]
                },
                {
                 'preProc':preProcParams,
                 'classify':[svm.SVC(probability=True,max_iter=500000,random_state=0)],
                 'classify__C': [1, 10, 100,1000], 
                 'classify__kernel': ['linear']
                },
                {
                 'preProc':preProcParams,
                 'classify':[svm.SVC(probability=True,max_iter=500000,random_state=0)],
                 'classify__C': [1, 10, 100,1000], 
                 'classify__kernel': ['rbf','sigmoid'],
                 'classify__gamma':[0.01,0.001,0.0001]
                },
                {
                 'preProc':preProcParams,
                 'classify':[svm.SVC(probability=True,max_iter=500000,random_state=0)],
                 'classify__C': [1, 10, 100,1000], 
                 'classify__kernel': ['poly'],
                 'classify__gamma':[0.01,0.001,0.0001],
                 'classify__degree':[2,3,4,5]
                },
                {
                  'preProc':preProcParams,
                  'classify':[RandomForestClassifier(random_state=0)],
                  "classify__n_estimators":[5,10,20,50,100],
                  "classify__criterion":["gini","entropy"],
                  "classify__max_features":["sqrt","log2",None],
                  "classify__max_depth":[None,2,3,5,8]
                        #"min_samples_split":[2,3,5,10,20],
                        #"min_samples_leaf":[1,2,3,5,10,20]
                },
                {
                    'preProc':preProcParams,
                    'classify':[tree.DecisionTreeClassifier()],
                    "classify__criterion":["gini","entropy"],
                    "classify__max_features":["sqrt","log2",None],
                    "classify__max_depth":[None,2,3,5,8],
                    "classify__min_samples_split":[2,3,5,10,20]
                },
                {
                        'preProc':preProcParams,
                        'classify':[MLPClassifier(random_state=0)],
                        "classify__activation":["identity","logistic","tanh","relu"],
                        "classify__solver":["lbfgs","sgd","adam"],
                        "classify__alpha":[0.01,0.001,0.0001,0.00001],
                        "classify__max_iter":[50,100,200,500]
                },
                {
                        'preProc':preProcParams,
                        'classify':[GaussianNB()]
                }
                ]
        pipeline=Pipeline([('scaling',StandardScaler()),('preProc',None),('classify',None)])
        #print("PARAMS")
        #print(pipeline.get_params())
        #time.sleep(5)
        classifier=GridSearchCV(pipeline, scoring = 'roc_auc',cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=0),verbose=10,param_grid=aParam_grid,n_jobs=4)
        #print("FIT")
        #time.sleep(5)
        classifier.fit(X,y)
        print(classifier.best_params_)
        print(classifier.best_score_)
        my_dict=classifier.cv_results_
        csvFile=open("TheVeryLongTestCicadaNew.csv","a")
        paramFile=open("TheVeryLongTestCicadaParamsNew.csv","a")
        for k in range(0,np.size(my_dict["mean_test_score"])):
            csvFile.write(str(i)+","+str(j)+","+str(k)+","+str(my_dict["mean_test_score"][k])+","+str(my_dict["std_test_score"][k])+","+str(my_dict['mean_fit_time'][k])+"\n")
            paramFile.write(str(i)+","+str(j)+","+str(k)+","+str(my_dict["params"][k])+"\n")
        csvFile.close()
        paramFile.close()
