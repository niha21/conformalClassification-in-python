#############################################
# ICP: Inductive Conformal Prediction
#        for Classification using RF
#############################################
# Import models from scikit learn module:
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sys


def computeConformityScores(modelFit,  calibrationData, calibrationTarget):

    if(modelFit is None) or (calibrationData is None):
        sys.exit("\n NULL model \n")

    nrCases, nrFeatures = calibrationData.shape
    #tempFrame = calibrationSet.loc[:, calibrationSet.columns[2:nrFeatures]]

    predProb = modelFit.predict_proba(calibrationData)
    nrCases, nrLabels = predProb.shape

    #nrLabels = len(predProb[1, :])  # number of class labels
    calibLabels = pd.to_numeric(calibrationTarget)

    MCListConfScores = []  # Moderian Class wise List of conformity scores
    for i in range(0, nrLabels):
        clsIndex = np.where(calibLabels == i)
        classMembers = predProb[clsIndex, i]
        MCListConfScores.append([classMembers])

    return(MCListConfScores)


def computePValues(MCListConfScores, testConfScores):

    if (MCListConfScores is None) or (testConfScores is None):
        sys.exit("\n NULL model \n")

    nrTestCases, nrLabels = testConfScores.shape
    pValues = np.zeros((nrTestCases,  nrLabels))

    for k in range(0, nrTestCases):
        for l in range(0, nrLabels):
            alpha = testConfScores[k, l]
            classConfScores = np.ndarray.flatten(np.array(MCListConfScores[l]))
            pVal = len(classConfScores[np.where(classConfScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(classConfScores[np.where(classConfScores == alpha)]))
            tempLen = len(classConfScores)
            pValues[k, l] = pVal/(tempLen + 1)

    return(pValues)


def ICPClassification(trainData, trainTarget, testData, ratioTrain=0.7, method="rf", nrTrees=100):
    if (trainData is None) or (testData is None):
        sys.exit("\n 'trainingSet' and 'testSet' are required as input\n")

    nrTrainCases, nrFeatures =  trainData.shape

    # create partition for proper-training set and calibration set.

    idx = np.random.permutation(nrTrainCases)
    properTrain = idx[:int(idx.size*ratioTrain)]
    calib = idx[int(idx.size*ratioTrain):]

    properTrainData = trainData[properTrain, :]
    properTrainTarget = trainTarget[properTrain]
    calibData = trainData[calib, :]
    calibTarget = trainTarget[calib]

    model = RandomForestClassifier(n_estimators=nrTrees)
    model.fit(properTrainData , properTrainTarget)
    #if(modelFit is None):
     #   return(None)

    MCListConfScores = computeConformityScores(model, calibData, calibTarget)
    testConfScores = model.predict_proba(testData)
    pValues = computePValues(MCListConfScores, testConfScores)

    return(pValues)



