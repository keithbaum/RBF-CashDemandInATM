from collections import namedtuple
import numpy as np

trainingAndValidationSet = namedtuple('trainingAndValidationSet','trainingSet validationSet')
dataSet=namedtuple('dataSet','datapoints results')

def normalize(datapoints):
    maximums=np.max(datapoints,axis=0)
    minimums=np.min(datapoints,axis=0)
    means=(maximums+minimums)/2.0
    deltas=maximums-minimums
    return (datapoints-means)/deltas


def setTrainingAndValidationSets(datapoints, results, percentageForTraining=0.8, iterations=None):
    '''Iterations based on cross-validation'''
    sampleDimension=len(results)
    trainingDimension=int(percentageForTraining*sampleDimension)
    iterations = iterations or 10
    shiftForIteration=int((sampleDimension-trainingDimension)/(iterations-1))
    result = []

    for i in range(iterations):
        trainingSelection=np.arange(i*shiftForIteration,i*shiftForIteration+trainingDimension)
        validationSelection=np.array(list(set(range(sampleDimension))-set(trainingSelection)))
        result.append( trainingAndValidationSet(dataSet(datapoints[trainingSelection,:],results[trainingSelection]),
                                                dataSet(datapoints[validationSelection, :], results[validationSelection])) )
    return result