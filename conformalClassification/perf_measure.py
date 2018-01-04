import matplotlib.pyplot as plt
import numpy as np
import sys
import math

# Computes efficiency of a conformal predictor, which is defined as the
# ratio of predictions with more than one class over the size of the testset
# @param matPValues Matrix of p-values
# @param testLabels True labels for the test-set
# @param sigfLevel Significance level
# @return The efficiency
# @export
def Efficiency(matPValues, testLabels, sigfLevel = 0.05):
    if (matPValues is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")


    nrTestCases = len(testLabels) #size of the test set
    nrLabels = len(set(testLabels))
    signifTest = np.ones((nrTestCases, nrLabels))
    err = 0

    for j in range(0, nrTestCases):
        for k in range(0, nrLabels):
            if matPValues[j,k] > sigfLevel:
                signifTest[j, k] = 1
            else:
                signifTest[j, k] = 0


    for j in range(0, nrTestCases):
        if sum(signifTest[j, :]) > 1:
            err = err + 1

    result = err/nrTestCases

    return(result)


# Computes error rate of a conformal predictor, which is defined as
# the ratio of predictions with missing true class lables over the size of the testset
# @param matPValues Matrix of p-values
# @param testLabels True labels for the test-set
# @param sigfLevel Significance level
# @return The error rate
# @export
def ErrorRate(matPValues, testLabels, sigfLevel = 0.05):
    if (matPValues is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")

    nrTestCases = len(testLabels)  # size of the test set
    nrLabels = len(set(testLabels))
    signifTest = np.ones((nrTestCases, nrLabels))

    for j in range(0, nrTestCases):
        for k in range(0, nrLabels):
            if matPValues[j, k] > sigfLevel:
                signifTest[j, k] = 1
            else:
                signifTest[j, k] = 0

    err = 0
    for j in range(0, nrTestCases):
        if signifTest[j, testLabels[j]] == 0:
            err = err + 1

    result = err / nrTestCases

    return result


# Computes observed fuzziness, which is defined as
# the sum of all p-values for the incorrect class labels.
# @param matPValues Matrix of p-values
# @param testLabels True labels for the test-set
# @return The observed fuzziness
# @export
def ObsFuzziness(matPValues, testLabels):
    if (matPValues is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")

    nrTestCases = len(testLabels)  # size of the test set
    sumPValues = 0

    for j in range(0, nrTestCases):
        sumPValues = sumPValues + sum(np.delete(matPValues[j, :], testLabels[j]))

    result = sumPValues / nrTestCases

    return result


# Computes the deviation from exact validity as the Euclidean norm of
# the difference of the observed error and the expected error
# @param matPValues Matrix of p-values
# @param testLabels True labels for the test-set
# @return The deviation from exact validity
# @export
def Validity(matPValues, testLabels):
    if (matPValues is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")

    nrTestCases = len(testLabels)
    nrLabels = len(set(testLabels))

    # compute error rate for the range of significance levels
    sigLevels = np.linspace(0, .99, 100)
    errAtSignif = np.zeros(len(sigLevels))
    signifTest = np.ones((nrTestCases, nrLabels))
    err = 0

    for i in range(1, len(sigLevels)):
        for j in range(0, nrTestCases):
            for k in range(0, nrLabels):
                if matPValues[j, k] > sigLevels[i]:
                    signifTest[j, k] = 1
                else:
                    signifTest[j, k] = 0

        err = 0
        for j in range(0, nrTestCases):
            if signifTest[j, testLabels[j]] == 0:
                err = err + 1

        err = err / nrTestCases
        errAtSignif[i] = pow((err - sigLevels[i]), 2)

    result = math.sqrt(sum(errAtSignif))

    return result


# Plots the calibration plot
# @param pValues Matrix of p-values
# @param testSet The test set
# @param color colour of the calibration line
# @return NULL
# @export
def CalibrationPlot(pValues, testLabels, color='b'):
    if (pValues is None) or (testLabels is None):
        sys.exit("\n 'pValues' and 'testLabels' are required as input\n")

    nrTestCases = len(testLabels)
    nrLabels = len(set(testLabels))

    #compute error rate for the range of significance levels
    sigLevels = np.linspace(0, .99, 100)
    errorRate = np.zeros(len(sigLevels))
    signifTest = np.ones((nrTestCases, nrLabels))
    for i in range(1, len(sigLevels)):
        for j in range(0, nrTestCases):
            for k in range(0, nrLabels):
                if pValues[j,k] > sigLevels[i]:
                    signifTest[j, k] = 1
                else:
                    signifTest[j, k] = 0

        err = 0
        for j in range(0, nrTestCases):
            if signifTest[j, testLabels[j]] == 0:
                err = err + 1

        errorRate[i] = err/nrTestCases


    #plot(sigLevels, errorRate, type = "l", xlab = "significance", ylab = "error rate",
       #col = color)
    plt.plot(sigLevels, errorRate, color=color)

    # draw diagonal line from (0,0) to (1,1)
    plt.annotate("",
                 xy=(0, 0), xycoords='data',
                 xytext=(1, 1), textcoords='data',
                 arrowprops=dict(arrowstyle="-",
                                 connectionstyle="arc3,rad=0.",
                                 color='r')
                 )
    plt.show()



#Function: pValues2PerfMetrics
#Desc: Computes performance measures: efficiency and validity of the conformal predictors
#input: matrix of p-values, test set and sifnificance level
#output: validity, efficiency, error rate and observed fuzziness
def pValues2PerfMetrics(matPValues, testLabels, sigfLevel = 0.05):
  errRate = ErrorRate(matPValues, testLabels, sigfLevel)
  eff = Efficiency(matPValues, testLabels, sigfLevel)
  val = Validity(matPValues, testLabels)
  obsFuzz = ObsFuzziness(matPValues, testLabels)

  print('Error rate: ', errRate)
  print('Efficiency: ', eff)
  print('Deviation from validity: ', val)
  print('Observed fuzziness: ', obsFuzz)

