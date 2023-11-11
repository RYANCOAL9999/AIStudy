import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#from sklearn import datasets - ########## Using IRIS data to replace ##########
df_iris = pd.read_csv('../dataSet/iris.csv')

df_iris.info()

df_iris.describle()

sns.set(style="darkgrid")

warnings.filterwarnings("ignore")

#%matplotlib inline - ####### using sns to show matplotlib with inline #######
sns.pairplot(df_iris)

wcss = []

x = df_iris.iloc[
        :, [
            0,
            1,
            2,
            3
        ]
    ].values

y_Kmeans = None

for i in range(1, 11):
    kmeans = KMeans(
       n_clusters= i, 
       init = 'k-means++',
       max_iter= 300,
       n_init = 10,
       random_state = 0 
    )
    y_Kmeans = kmeans.fit_predict(x) 
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('number of cluster')
plt.ylabel('WCSS')
plt.show()

plt.scatter(
    x[y_Kmeans == 0, 0], 
    x[y_Kmeans == 0, 1], 
    s = 100, 
    c = 'red', 
    label = 'Iris-setosa'
)

plt.scatter(
    x[y_Kmeans == 1, 0], 
    x[y_Kmeans == 1, 1], 
    s = 100, 
    c = 'blue', 
    label = 'Iris-versicolour'
)

plt.scatter(
    x[y_Kmeans == 2, 0], 
    x[y_Kmeans == 2, 1], 
    s = 100, 
    c = 'green', 
    label = 'Iris-virginica'
)

plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1],  
    s = 100, 
    c = 'yellow', 
    label = 'Centroids'
)

plt.legend()
