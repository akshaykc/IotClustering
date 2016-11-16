'''
Created on Nov 15, 2016

@author: achaluv
'''
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering

data=[]
with open('achaluv_clustering_data.txt','r') as f:
    num = 0
    for line in f:
        nums = line.split(',')
        data.append([float(nums[0]),float(nums[1]),float(nums[2])])
    data = np.array(data)
    
    
    
    
    
    
    
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

def eblow(df, n):
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df)**2)/df.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()
    
eblow(data,40)

clusters = KMeans(n_clusters=2).fit(data)
centroids = clusters.cluster_centers_
label = clusters.labels_


fig = plt.figure()
ax = p3.Axes3D(fig)
for l in np.unique(label):
    ax.plot3D(data[label == l, 0], data[label == l, 1], data[label == l, 2],
              'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)))
plt.title('KMeans clusters')
plt.show()
