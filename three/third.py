import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保字体支持
plt.rcParams['axes.unicode_minus'] = False    # 确保负号显示正常

# 读取数据集
df = pd.read_csv('US-pumpkins.csv')

print('数据基本信息：')
df.info()

# 数据描述统计
print("\n数据的描述性统计：")
print(df.describe())

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 处理缺失值，这里简单填充为 0
df = df.fillna(0)

# 提取特征变量和目标变量
X = df[['Low Price', 'High Price', 'Mostly High']]
y = df['Mostly Low']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差：{mse}')
print(f'平均绝对误差：{mae}')
print(f'R² 分数：{r2}')

# 交叉验证
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f'交叉验证得分：{cv_scores}')
print(f'平均交叉验证得分：{cv_scores.mean()}')

# 设置图片清晰度，适当降低
plt.rcParams['figure.dpi'] = 150  # 将DPI设置为150，以避免图片过大

# 可视化预测结果
plt.figure(figsize=(8, 6))  # 调整图片大小
plt.scatter(y_test, y_pred)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('线性回归模型预测结果可视化')  # 确保中文显示正常
plt.xticks(rotation=45)
plt.show()

# 绘制特征的分布直方图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 可以减小figsize
for i, col in enumerate(X.columns):
    sns.histplot(X[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f'{col} 分布')
plt.tight_layout()
plt.show()

# 绘制特征与目标变量的散点图矩阵
g = sns.pairplot(df[['Low Price', 'High Price', 'Mostly High', 'Mostly Low']])
g.fig.suptitle('特征与目标变量的散点图矩阵', y=1.02)
plt.show()

# 计算残差
residuals = y_test - y_pred

# 绘制残差图
plt.figure(figsize=(8, 6))  # 调整图片大小
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('线性回归模型残差图')
plt.show()

# 绘制残差的分布图
plt.figure(figsize=(8, 6))  # 调整图片大小
sns.histplot(residuals, kde=True, bins=20)
plt.title('残差分布')
plt.xlabel('残差值')
plt.ylabel('频率')
plt.show()
