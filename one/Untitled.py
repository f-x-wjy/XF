import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 加载数据集
df = pd.read_csv('nigerian-songs.csv')

# 查看数据集的基本信息
print(df.info())

# 处理缺失值（这里我们使用均值插补作为示例）
imputer = SimpleImputer(strategy='mean')
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# 标准化数据
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

from sklearn.cluster import KMeans

# 选择用于聚类的特征（这里我们使用所有数值特征）
X = df[numerical_cols]

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 查看聚类结果
print(df.groupby('cluster').mean())

from sklearn.cluster import SpectralBiclustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 创建一个示例数据矩阵（这里我们使用原始数据的子集作为示例）
data_matrix = X.values

# 应用SpectralBiclustering
model = SpectralBiclustering(n_clusters=(3, 2), random_state=0)
model.fit(data_matrix)

# 绘制层次聚类树状图（仅作为示例，因为SpectralBiclustering不直接提供层次结构）
# 这里我们使用linkage和dendrogram函数来演示如何绘制树状图，但实际应用中可能需要其他方法
linked = linkage(data_matrix, 'single')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.show()

import pandas as pd

# 加载数据集
df = pd.read_csv("nigerian-songs.csv")

# 查看数据概览
print(df.head())

# 检查缺失值
print(df.isnull().sum())

# 处理缺失值，填补缺失的数值列
df.fillna(df.mean(), inplace=True)

# 转换数值列的类型
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

from sklearn.preprocessing import StandardScaler

# 标准化数值列
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']])

# 将标准化后的数据转回数据框
scaled_df = pd.DataFrame(scaled_features, columns=['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])

from sklearn.cluster import KMeans

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_df)

# 查看聚类结果
print(df[['name', 'artist', 'cluster']].head())

from sklearn.cross_decomposition import PLSRegression

# 假设我们想预测 'popularity'，因此将 'popularity' 列作为目标
X = scaled_df.drop('popularity', axis=1)
y = scaled_df['popularity']

# 进行交叉分解
pls = PLSRegression(n_components=2)
pls.fit(X, y)

# 查看交叉分解的结果
print(pls.x_scores_)  # 解释变量的主成分

from sklearn.impute import KNNImputer

# 使用 KNNImputer 填补缺失值
imputer = KNNImputer(n_neighbors=2)
df_imputed = imputer.fit_transform(df[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']])

# 将填补后的数据转回数据框
df_imputed = pd.DataFrame(df_imputed, columns=['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])

import matplotlib.pyplot as plt

# 绘制 K-means 聚类的散点图
plt.scatter(df['danceability'], df['energy'], c=df['cluster'], cmap='viridis')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('K-means Clustering Results')
plt.show()






