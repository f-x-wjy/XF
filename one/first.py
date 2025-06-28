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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据
file_path = r'C:\Users\86185\Desktop\新建文件夹\nigerian-songs.csv'
df = pd.read_csv(file_path)

# 初步查看数据
print("数据前五行：")
print(df.head())
print("\n数据基本信息：")
print(df.info())
print("\n统计摘要：")
print(df.describe())
print("\n缺失值情况：")
print(df.isnull().sum())


# 处理缺失值 - 根据实际情况选择删除或填充
df = df.dropna()  # 简单处理：删除含有缺失值的行
# 或者用均值/中位数填充数值列
# df.fillna(df.mean(), inplace=True)

# 删除不必要的列（如果有）
# df = df.drop(['列名1', '列名2'], axis=1)

# 检查并处理重复值
df = df.drop_duplicates()

# 检查异常值
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title(f'{col} 箱线图')
    plt.show()

    # 特征编码 - 处理分类变量
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 特征缩放 - 标准化数值特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select_dtypes(include=['int64', 'float64']))
    df_scaled = pd.DataFrame(scaled_features, columns=df.select_dtypes(include=['int64', 'float64']).columns)

    # 检查相关性
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
    plt.title('特征相关性热图')
    plt.show()


# 确定最佳聚类数 - 肘部法则
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('聚类数')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.show()

# 根据肘部法则选择k值（假设k=4）
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# 评估聚类效果
silhouette_avg = silhouette_score(df_scaled, clusters)
print(f"轮廓系数: {silhouette_avg}")

# 将聚类结果添加到原始数据
df['Cluster'] = clusters

# 可视化聚类结果（选择两个主要特征）
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=clusters, cmap='viridis')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('K-Means聚类结果')
plt.show()


# 准备特征和目标变量
X = df_scaled.drop('popularity', axis=1)  # 假设'popularity'是目标变量
y = df['popularity']  # 使用原始值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差(MSE): {mse}")
print(f"R平方值: {r2}")

# 可视化预测结果
plt.scatter(y_test, y_pred)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# 特征重要性
coef = pd.Series(lr.coef_, index=X.columns)
coef.sort_values(ascending=False).plot(kind='barh')
plt.title('特征重要性')
plt.show()
