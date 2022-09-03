import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing mall data set with pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# Using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidean Dist')
plt.show()

# Fitting hierarchical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)

# Visualising clusters
plt.scatter(x[y_hc==0,0], x[y_hc==0,1],s=100,c='red',label='Careful',edgecolors='black')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1],s=100,c='royalblue',label='Standard',edgecolors='black')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1],s=100,c='green',label='Target',edgecolors='black')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1],s=100,c='skyblue',label='Careless',edgecolors='black')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1],s=100,c='magenta',label='Sensible',edgecolors='black')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(K$')
plt.ylabel('Spending Score(1-100')
plt.legend()
plt.show()
