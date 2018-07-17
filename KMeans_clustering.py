import numpy as np
import datetime
from functools import lru_cache
from matplotlib import pyplot as plt
from collections import namedtuple
from data import normalize, setTrainingAndValidationSets

clusterWithGaussian = namedtuple('clusterWithGaussian','mu sigma cluster')

class singlePoint(object):
    def __init__( self, coordinates=0 ):
        self.coordinates = np.array( coordinates )

    @property
    def dimensions(self):
        return self.coordinates.shape

    @property
    def norm(self):
        return np.sqrt( np.sum( np.power( self.coordinates, 2 ) ) )

    def __sub__(self, otherPoint):
        return singlePoint( self.coordinates-otherPoint.coordinates )

    def __rsub__(self, otherPoint):
        return self.__sub__(otherPoint)

    def __add__(self, otherPoint):
        return singlePoint( self.coordinates + otherPoint.coordinates )

    def __radd__(self, otherPoint):
        return self.__add__(otherPoint)

    def __pow__(self, power, modulo=None):
        return singlePoint( np.power( self.coordinates, power) )

    def __truediv__(self, number):
        if number==0:
            return singlePoint( np.ones(self.dimensions)*np.inf )
        return singlePoint( self.coordinates/number )

    def distance(self, otherPoint=None):
        otherPoint = otherPoint or singlePoint( np.zeros( self.dimensions ) )
        return ( self - otherPoint ).norm

class guassianNode(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self,point):
        return np.exp(-point.distance(self.mu)**2/(2*self.sigma))

class KMeans (object):
    def __init__(self,X, nClusters=9, tolerance=1e-3, maxIterations=50):
        self.points = X
        self.nClusters = nClusters
        self.tolerance = tolerance
        self.maxIterations = maxIterations
        self._clusterize()

    def _clusterize(self):
        clusters = self._getCenters()
        sigmas = self._getSigmas(clusters)
        self.clustersWithGaussians = [ clusterWithGaussian(mu,sigmas[mu],clusters[mu]) for mu in list(clusters.keys()) ]

    def _getCenters(self):
        newMus = np.random.choice(self.points,self.nClusters)
        error = np.inf
        repetitions=0
        while( error > self.tolerance and repetitions<self.maxIterations):
            mus = newMus
            clusters = { mu:[] for mu in mus }
            for x in self.points:
                distances = [mu.distance(x) for mu in mus]
                argmin = distances.index( min( distances ) )
                clusters[ mus[ argmin ] ].append( x )
            newMus = [ sum( cluster,singlePoint() )/len(cluster) for cluster in  clusters.values() ]
            error = max( [ mu.distance(newMu) for mu,newMu in zip(mus,newMus) ] )
            repetitions+=1
        if repetitions==self.maxIterations:
            print("Didn't converge. Reached max iterations. Error=%.5f"%error)
        return clusters

    @classmethod
    def _getSigmas(cls, clusters):
        mus = list( clusters.keys() )
        result={}
        for mu in mus:
            if len(clusters[mu])==0:
                result[mu]=0
                continue
            result[mu]=np.sqrt( sum( [ point.distance(mu)**2 for point in clusters[mu] ] )/len(clusters[mu] ) )
        return result

    @property
    def clusters(self):
        return {aClusterWithGaussian.mu: aClusterWithGaussian.cluster for aClusterWithGaussian in self.clustersWithGaussians}

    @property
    @lru_cache(maxsize=None)
    def RBFMatrix(self):
        rbfMatrix = np.empty((len(self.points),self.nClusters))
        for j,gaussianFunction in enumerate(self.RBFSet):
            for n,point in enumerate(self.points):
                rbfMatrix[n,j]=gaussianFunction.evaluate(point)
        return rbfMatrix

    @property
    @lru_cache(maxsize=None)
    def RBFSet(self):
        return [ guassianNode(cluster.mu, cluster.sigma) for cluster in self.clustersWithGaussians ]

    def weightsFromTraining(self,givenResults):
        if isinstance(givenResults, list):
            givenResults=np.array(givenResults)
        pseudoInverseRBFMatrix=np.linalg.inv(self.RBFMatrix.transpose()@self.RBFMatrix)@self.RBFMatrix.transpose()
        weights=pseudoInverseRBFMatrix@givenResults
        return weights

    def evaluate(self,weights,points):
        result=np.zeros(len(points))
        for n,point in enumerate(points):
            for j, gaussianFunction in enumerate(self.RBFSet):
                result[n]+=weights[j]*gaussianFunction.evaluate(point)
        return result

    def plotClusters(self,weights=None):
        mus = list ( self.clusters.keys() )
        if mus[0].dimensions[0]>2:
            print("I cannot plot this!")
            return
        color = iter(plt.get_cmap('rainbow')(np.linspace(0, 1, len(mus))))
        for i,mu in enumerate(mus):
            points = np.array( [ x.coordinates for x in self.clusters[mu] ] )
            c=next(color)
            plt.scatter(points[:, 0], points[:, 1], color=c, alpha=0.3)
            plt.scatter( mu.coordinates[0],mu.coordinates[1] , color=c, s=200, marker='X')
            if weights is not None:
                legend = 'sigma=%.2f\nweight=%.2f' % (self.clustersWithGaussians[i].sigma,
                                                  weights[i])
                plt.annotate(legend, (mu.coordinates[0], mu.coordinates[1]))
        plt.show()


def calculateRSS(krange=None, percentageForTraining=0.8):
    RSS=[]
    krange=krange or range(2,15)
    for i in krange:
        print('Testing for %s clusters'%i)
        np.random.seed(datetime.datetime.now().microsecond)
        datasets = setTrainingAndValidationSets(datapoints, results, percentageForTraining)
        trainingPoints = [singlePoint(datasets.trainingSet.datapoints[i, :]) for i in
                          range(datasets.trainingSet.datapoints.shape[0])]
        validationPoints = [singlePoint(datasets.validationSet.datapoints[i, :]) for i in
                            range(datasets.validationSet.datapoints.shape[0])]

        kmeans = KMeans(trainingPoints,nClusters=i)
        weights=kmeans.weightsFromTraining(datasets.trainingSet.results)
        estimatedResults=np.sign( kmeans.evaluate(weights,validationPoints) )*0.5
        rss=0.5*np.sum( np.power(estimatedResults-datasets.validationSet.results,2))
        RSS.append( rss )
        print("RSS=%.5f"%rss)
    return RSS

data=np.loadtxt('data.csv',int, delimiter=',',skiprows=1,usecols=range(1,22))
results= normalize(data[:, -1])
datapoints= normalize(data[:, :-1])

krange=range(2,15)
RSS=calculateRSS(krange,0.8)
plt.plot(krange,RSS)
plt.show()

# plt.plot(datasets.validationSet.results,c='blue',label='Original')
# plt.plot(estimatedResults,c='green',label='Estimated')
# plt.legend()


#kmeans.plotClusters(weights)
