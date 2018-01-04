#############################################
# TCP: Transductive Conformal Prediction
#        for Classification using RF
#############################################
# Import models from scikit learn module:
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
# Note: This function is internal to the package, but global
# since it is shared across threads for parallel computing
# Fits the model and computes p-values, internal to the package
# @param augTrainSet Augmented training set
# @param method Method for modeling
# @param nrTrees Number of trees for RF
# @return The p-values


def tcpPValues(tcpTrainData, tcpTrainTarget, method="rf", nrTrees=100):
    model = RandomForestClassifier(n_estimators=nrTrees)

    modelFit = model.fit(tcpTrainData, tcpTrainTarget)
    if(modelFit is None):
        sys.exit("\n NULL model \n")

    testLabel = tcpTrainTarget[len(tcpTrainTarget)-1]  # test case class label

    # consider only those training samples that are labelled as testLabel
    classSamples = np.where(tcpTrainTarget == testLabel)[0]

    predProbability = model.predict_proba(tcpTrainData[classSamples, :])

    nonconformityScores = predProbability[:, testLabel]

    # test case prediction probability
    alpha = predProbability[len(classSamples)-1, testLabel]
    pVal = len(nonconformityScores[np.where(nonconformityScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(nonconformityScores[np.where(nonconformityScores == alpha)]))
    pVal = pVal / (len(nonconformityScores))

    return(pVal)

def TCPClassification(trainData, trainTarget, testData, ratioTrain=0.7, method="rf", nrTrees=100):
    if (trainData is None) or (testData is None):
        sys.exit("\n 'trainingSet' and 'testSet' are required as input\n")

    nrTestCases, nrFeatures = testData.shape

    nrLabels = len(np.unique(trainTarget))
    pValues = np.zeros((nrTestCases, nrLabels))

    for i in range(0, nrLabels):
        clsLabel = np.array([i])
        for k in range(0, nrTestCases):
            tcpTrainTarget = np.concatenate((trainTarget, clsLabel))
            tcpTrainData = np.append(trainData, [testData[k, :]], axis = 0)
            pValues[k, i] = tcpPValues(tcpTrainData, tcpTrainTarget, method=method, nrTrees=nrTrees)

    return(pValues)