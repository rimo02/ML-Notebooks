import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv('/Users/KIIT/Machine Learning/Clustering/OnlineRetail.csv',
                 encoding='ISO-8859-1', sep=",", header=0)
df.head()
df.info()
df.describe()
df.isnull().sum()
df.columns
df['Amount'] = df['Quantity'] * df['UnitPrice']

df = df.dropna()

print(len(df['CustomerID'].unique()))
print(df.shape)


# We are going to cluster based on the following factors
# 1- total amount from each indiviual customer
# 2- frequency of invoice for each customer

df_amnt = df.groupby('CustomerID')['Amount'].sum()
df_amnt = df_amnt.reset_index()
df_amnt.head()


df_freq = df.groupby('CustomerID')['InvoiceNo'].count()
df_freq = df_freq.reset_index()
print(df_freq.head())

retail_df = pd.merge(df_amnt, df_freq, on='CustomerID', how='inner')
retail_df.head()
retail_df.info()

plt.figure(figsize=(10, 8))
sns.boxplot(data=retail_df[['Amount', 'InvoiceNo']], orient="v",
            palette="Set2", whis=1.5, saturation=1, width=0.7)
plt.title("Outlier distribution")
plt.xlabel("Attribues")
plt.ylabel("Range")
plt.show()

# Removing outliers for Amount
Q1 = retail_df.Amount.quantile(0.05)
Q3 = retail_df.Amount.quantile(0.95)
inter_quartile_range = Q3 - Q1
print(inter_quartile_range)
retail_df = retail_df[(retail_df.Amount >= Q1 - 1.5 * inter_quartile_range)
                      & (retail_df.Amount <= Q3 + 1.5*inter_quartile_range)]
retail_df


# Removing outliers for Frequency
Q1 = retail_df.InvoiceNo.quantile(0.05)
Q3 = retail_df.InvoiceNo.quantile(0.95)
inter_quartile_range = Q3 - Q1
print(inter_quartile_range)
retail_df = retail_df[(retail_df.InvoiceNo >= Q1 - 1.5 * inter_quartile_range)
                      & (retail_df.InvoiceNo <= Q3 + 1.5*inter_quartile_range)]
retail_df

# Scaling
scaler = StandardScaler()
scaled_df = scaler.fit_transform(retail_df)
scaled_df

# Single Linkage
plt.figure(figsize=(12, 10))
mergings1 = linkage(scaled_df, method="single", metric="euclidean")
print(mergings1)
dendrogram(mergings1)
plt.title("Single Linkage")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Complete Linkage
plt.figure(figsize=(12, 10))
mergings2 = linkage(scaled_df, method="complete", metric="euclidean")
print(mergings2)
dendrogram(mergings2)
plt.title("Complete Linkage")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Aberage Linkage
plt.figure(figsize=(12, 10))
mergings3 = linkage(scaled_df, method="average", metric="euclidean")
print(mergings3)
dendrogram(mergings3)
plt.title("Average Linkage")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Cutting the dendogram based on df values
dbi_k_score = []
si_score = []
for k in range(2, 5):
    cluster_labels = cut_tree(mergings3, n_clusters=k).reshape(-1,)
    dbi_k_score.append(davies_bouldin_score(retail_df, cluster_labels))
    si_score.append(silhouette_score(retail_df, cluster_labels))

# Optimal value of K = 3

cluster_labels = cut_tree(mergings3, n_clusters=3).reshape(-1,)
retail_df['Cluster'] = (cluster_labels)
retail_df.sample()

print(retail_df['Cluster'].unique())

plt.figure(figsize=(12, 8))
sns.boxplot(data=retail_df, x='Cluster', y='Amount')
plt.xlabel('Cluster')
plt.ylabel("Amount")
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(data=retail_df, x='Cluster', y='InvoiceNo')
plt.xlabel('Cluster')
plt.ylabel("Frequency")
plt.show()

silhoute_score = silhouette_score(retail_df, cluster_labels)
dbi_score = davies_bouldin_score(retail_df, cluster_labels)
print("Silhoute Score = ", silhoute_score)
print("Dbi Score = ", dbi_score)
