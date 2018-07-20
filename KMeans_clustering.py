import numpy as np
import matplotlib
import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
from matplotlib import pyplot as plt
from collections import namedtuple

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

    def __eq__(self, otherPoint):
        return np.array_equal(self.coordinates,otherPoint.coordinates)

    def __hash__(self):
        return hash(id(self))

    def distance(self, otherPoint=None):
        otherPoint = otherPoint or singlePoint( np.zeros( self.dimensions ) )
        return ( self - otherPoint ).norm

class guassianNode(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self,point):
        if self.sigma==0:       #Then this is a delta function rather than a gaussian
            return 1 if point==self.mu else 0
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
        newMus = np.random.choice(self.points,self.nClusters,replace=False)
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
            error = max([mu.distance(newMu) for mu, newMu in zip(mus, newMus)])
            repetitions += 1
            for i,cluster in enumerate(clusters.values()):  #Look for empty clusters and fix it!
                if len(cluster)==0:
                    print('Found empty cluster! Resetting clusterization')
                    newMus[i]=np.random.choice(self.points,1)[0]
                    repetitions=0
                    error=np.inf
            if repetitions == self.maxIterations:
                print("Didn't converge. Reached max iterations. Error=%.5f" % error)
        return clusters

    @classmethod
    def _getSigmas(cls, clusters):
        mus = list( clusters.keys() )
        result={}
        for mu in mus:
            result[mu]=np.sqrt( sum( [ point.distance(mu)**2 for point in clusters[mu] ] )/len(clusters[mu] ) )
        return result

    @property
    def clusters(self):
        return {aClusterWithGaussian.mu: aClusterWithGaussian.cluster for aClusterWithGaussian in self.clustersWithGaussians}

    @property
    def RBFMatrix(self):
        rbfMatrix = np.empty((len(self.points),self.nClusters))
        for j,gaussianFunction in enumerate(self.RBFSet):
            for n,point in enumerate(self.points):
                rbfMatrix[n,j]=gaussianFunction.evaluate(point)
        return rbfMatrix

    @property
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
