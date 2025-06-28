import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = r'C:\Users\86185\Desktop\新建文件夹\nigerian-songs.csv'
df = pd.read_csv(file_path)

# 初步查看数据
print(df.head())
print("\n数据基本信息：")
print(df.info())

# 处理缺失值和重复值
df = df.dropna().drop_duplicates()

# 可视化数值特征的箱线图，检查异常值
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(numeric_cols)  # 统计数值特征的数量
n_rows = (num_plots // 3) + (num_plots % 3 > 0)  # 计算行数
n_cols = 3  # 每行3列

plt.figure(figsize=(15, 5 * n_rows))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)  # 动态创建子图
    sns.boxplot(x=df[col])
    plt.title(f'{col} 箱线图')
plt.tight_layout()
plt.show()

# 特征编码与标准化
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['int64', 'float64'])),
                         columns=df.select_dtypes(include=['int64', 'float64']).columns)

# 可视化特征相关性
plt.figure(figsize=(12, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性热图')
plt.show()

# 聚类分析（KMeans & 肘部法则）
inertia = [KMeans(n_clusters=k, random_state=42).fit(df_scaled).inertia_ for k in range(2, 11)]
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('聚类数')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.show()

# 聚类结果
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df['Cluster'] = clusters

# 评估聚类效果
silhouette_avg = silhouette_score(df_scaled, clusters)
print(f"轮廓系数: {silhouette_avg}")

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=clusters, cmap='viridis')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('K-Means聚类结果')
plt.show()

# 线性回归模型
X = df_scaled.drop('popularity', axis=1)
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差(MSE): {mse}")
print(f"R平方值: {r2}")

# 预测结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()


# 特征重要性
coef = pd.Series(lr.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
coef.sort_values(ascending=False).plot(kind='barh')
plt.title('特征重要性')
plt.show()
