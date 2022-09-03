import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing mall data set with pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# Using elbow method to find optimum number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying kmeans to the mall dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualising clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1],s=100,c='red',label='Careful',edgecolors='black')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1],s=100,c='royalblue',label='Standard',edgecolors='black')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1],s=100,c='green',label='Target',edgecolors='black')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1],s=100,c='skyblue',label='Careless',edgecolors='black')
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1],s=100,c='magenta',label='Sensible',edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c = 'yellow', label='centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(K$')
plt.ylabel('Spending Score(1-100')
plt.legend()
plt.show()
