import datetime
import numpy as np
from KMeans_clustering import singlePoint, KMeans
from data import setTrainingAndValidationSets


def calculateRSS(datapoints, results, krange=None, percentageForTraining=0.8):
    RSS=[]
    krange=krange or range(2,15)
    for i in krange:
        RSS.append(_calculateRSSForNClusters(datapoints, results, nClusters=i,percentageForTraining=percentageForTraining))
    return RSS


def _calculateRSSForNClusters(datapoints, results, nClusters=None, percentageForTraining=0.8):
    nClusters=nClusters or 9
    np.random.seed(datetime.datetime.now().microsecond)
    datasets = setTrainingAndValidationSets(datapoints, results, percentageForTraining)
    print('Testing for %s clusters' % nClusters)
    rsss=[]
    for dataset in datasets:
        trainingPoints = [singlePoint(dataset.trainingSet.datapoints[i, :]) for i in
                          range(dataset.trainingSet.datapoints.shape[0])]
        validationPoints = [singlePoint(dataset.validationSet.datapoints[i, :]) for i in
                            range(dataset.validationSet.datapoints.shape[0])]

        kmeans = KMeans(trainingPoints, nClusters=nClusters)
        weights = kmeans.weightsFromTraining(dataset.trainingSet.results)
        estimatedResults = np.sign(kmeans.evaluate(weights, validationPoints)) * 0.5
        rsss.append( np.sum(np.power(estimatedResults - dataset.validationSet.results, 2)) / len(estimatedResults) * 100.0 )
    rss=np.mean(rsss)

    print("RSS=%.5f" % rss + '%')
    return rss