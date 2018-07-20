import matplotlib
import matplotlib.style
matplotlib.use("Qt5Agg")
matplotlib.style.use('classic')
from matplotlib import pyplot as plt

from rss import calculateRSS
from data import normalize
import numpy as np

data=np.loadtxt('data.csv',int, delimiter=',',skiprows=1,usecols=range(1,22))
results= normalize(data[:, -1])
datapoints= normalize(data[:, :-1])

krange=range(2,15)
percentegesForTraining=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
color = iter(plt.get_cmap('rainbow')(np.linspace(0, 1, len(percentegesForTraining))))
RSSs=np.empty((len(percentegesForTraining),len(krange)))
for i,percentageForTraining in enumerate(percentegesForTraining):
    print(str(percentageForTraining*100.0)+'% of samples for training')
    RSS= calculateRSS(datapoints, results, krange, percentageForTraining)
    RSSs[i,:]=RSS
    c = next(color)
    plt.plot(krange,RSS,label='training with '+str(percentageForTraining*100)+'%',color=c)
plt.plot(krange,np.mean(RSSs,axis=0),color='black',linestyle='--', linewidth=2,label='Average')
plt.legend()
plt.show()