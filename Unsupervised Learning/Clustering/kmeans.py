import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
df = pd.read_csv('/Users/KIIT/Machine Learning/Clustering/Mall_Customers.csv')

df.drop(['CustomerID'], axis=1, inplace=True)
df.info()
df.describe()
df.isnull().sum()

# %matplotlib inline
sns.pairplot(df)
# clearly we can cluster spending score vs annual income

# %matplotlib inline
plt.figure(figsize=(10, 10))
sns.scatterplot(x=df['Annual Income (k$)'],
                y=df['Spending Score (1-100)'], data=df, hue=df['Gender'], s=100)
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.show()

features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print(features)

inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

print(inertias)

# %matplotlib inline
plt.figure(figsize=(10, 8))
plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel("No of cluster (K")
plt.ylabel("Within Sum of squares of clusters")
plt.title("Elbow method for KMeans")

# Optimally we can segregate into 5 classes of customers
kmeans = KMeans(n_clusters=5, random_state=42)
df['Clusters'] = kmeans.fit_predict(features)

# %matplotlib inline
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Clusters', s=100, markers='o', palette="viridis")
plt.scatter(x=kmeans.cluster_centers_[:, 1], y=kmeans.cluster_centers_[
            :, 2], c='red', s=200, label='Centroids', marker='X')
plt.title("Kmeans clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

silhoute_score = silhouette_score(features, kmeans.labels_)
dbi_score = davies_bouldin_score(features, kmeans.labels_)
print("Silhoute Score = ", silhoute_score)
print("Dbi Score = ", dbi_score)
