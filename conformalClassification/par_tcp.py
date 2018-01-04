#############################################
# TCP: Transductive Conformal Prediction
#        for Classification using RF
#############################################
# Import models from scikit learn module:
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
import multiprocessing
import conformalClassification.config as config


#############################################
# TCP: Transductive Conformal Prediction
#        for Classification using RF
#############################################
# Note: This function is internal to the package, but global
# since it is shared across threads for parallel computing
# Fits the model and computes p-values, internal to the package
# @param augTrainSet Augmented training set
# @param method Method for modeling
# @param nrTrees Number of trees for RF
# @return The p-values


def parTCP(k):
    tcpTrainTarget = np.append(config.trainTarget, [config.clsLabel])
    tcpTrainData = np.append(config.trainData, [config.testData[k, :]], axis=0)
    model = RandomForestClassifier(n_estimators = config.nrTrees)

    modelFit = model.fit(tcpTrainData, tcpTrainTarget)
    if (modelFit is None):
        sys.exit("\n NULL model \n")

    testLabel = tcpTrainTarget[len(tcpTrainTarget) - 1]  # test case class label

    # consider only those training samples that are labelled as testLabel
    classSamples = np.where(tcpTrainTarget == testLabel)[0]

    predProbability = model.predict_proba(tcpTrainData[classSamples, :])

    nonconformityScores = predProbability[:, testLabel]

    # test case prediction probability
    alpha = predProbability[len(classSamples) - 1, testLabel]
    pVal = len(nonconformityScores[np.where(nonconformityScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                                                                              len(nonconformityScores[np.where(
                                                                                  nonconformityScores == alpha)]))
    pVal = pVal / (len(nonconformityScores))

    return (pVal)



def TCPClassification(trainData, trainTarget, testData, method="rf", nrTrees=100):
    if (trainData is None) or (testData is None):
        sys.exit("\n 'trainingSet' and 'testSet' are required as input\n")

    config.trainData = trainData
    config.trainTarget = trainTarget
    config.testData = testData
    config.nrTrees = nrTrees

    nrTestCases, nrFeatures = testData.shape

    nrLabels = len(np.unique(trainTarget))
    pValues = np.zeros((nrTestCases, nrLabels))

    for i in range(0, nrLabels):
        config.clsLabel = [i]
        pool = multiprocessing.Pool(processes=4)
        pValues[:, i] = pool.map(parTCP, range(0, nrTestCases))
        pool.close()
        pool.join()

    return(pValues)
