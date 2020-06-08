Source code used in developing the research work 'Automatic Rain and Cicada Chorus Filtering of Bird Acoustic Data' currently under review for the publication 'Applied Soft Computing.'

Requirements:

MMSE STSA processed using code by Alexander Chiu: https://github.com/alexanderchiu/AudioProcessor
High-Pass filter and spectrograms generated using SoX: http://sox.sourceforge.net/
Acoustic Features processed with the help of Apache Commons Math: http://commons.apache.org/proper/commons-math/

Process:
Upon applying chosen pre-processing to audio, execute WriteIndicesSet.java to calculate acoustic indices. This generates a .arff file which is formatted for use with WEKA. Outher Java included classes facilitate this process.

Classification testing is carried out by GridSearchClassifier.py which reinterprets WEKA files for use in SciKit-learn and applies a GridSearch. 

ROCPlot.py carries plots an ROC for a given classifier to determine accuracy

BigClassifierTest.py applies a classification model to a test set and outputs classification probabilities of each file as a CSV