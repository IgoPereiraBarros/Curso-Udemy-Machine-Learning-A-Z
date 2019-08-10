# -*- coding: utf-8 -*-

#%reset -f

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset with pandas
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values

# Importing the dendrogram to find the optimal number of clusters
''' 
Aqui precisamos encontrar o número ótimo de clusters, para isso precisamos consultar um
dendograma. E seguir o seguintes passos para encontrar o número de clusters ideais:

1 - Encontrar a maior distâcia da linha vertical e não cruzar qualquer linha horizontal
2 - Contar o número de linhas verticais, seguindo a primeira etapa.
'''
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
# Resultado = 5 clusters

# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualization clusters
plt.scatter(X[y_])