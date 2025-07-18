import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import graphviz

# 设置中文显示
plt.rcParams["font.family"] = ["SimSun-ExtB", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常
# 1. 读取数据
df = pd.read_csv('US-pumpkins.csv')

# 2. 数据预处理
# 2.1 填充缺失值（只对数值列填充）
numeric_columns = df.select_dtypes(include=[np.number]).columns  # 选择数值列
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# 2.2 衍生特征
df['Price_Diff'] = df['High Price'] - df['Low Price']

# 3. 相关性分析
numeric_columns = df.select_dtypes(include=[np.number])
corr_matrix = numeric_columns.corr()
print(corr_matrix)

# 3.2 打印与目标变量的相关系数
target = 'Mostly Low'
corr_target = corr_matrix[target].drop(target).sort_values(ascending=False)
print(f"目标变量与其他特征的相关系数：\n{corr_target}")

# 可视化相关矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('相关矩阵热图')
plt.show()

# 4. 特征选择
threshold = 0.3
selected_features = corr_target[abs(corr_target) > threshold].index.tolist()
selected_features.append('Price_Diff')  # 将衍生特征也加入
print(f"选择的特征（相关系数绝对值 > {threshold}）：{selected_features}")

# 5. 数据准备
X = df[selected_features]
y = df[target]

# 6. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 8. 模型训练与评估函数
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name}模型评估：")
    print(f"均方误差：{mse}")
    print(f"平均绝对误差：{mae}")
    print(f"R方 分数：{r2}")

    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }


# 9. 定义模型参数（可自定义修改）

rf_params = {
    'n_estimators': 10,  # 树的数量
    'max_depth': 4,  # 树的最大深度
    'min_samples_split': 100,  # 拆分内部节点所需的最小样本数
    'random_state': 42
}

lgb_params = {
    'objective': 'regression',
    'n_estimators': 20,  # 提升树的数量
    'learning_rate': 0.1,  # 学习率
    'max_depth': 8,  # 树的最大深度
    'num_leaves': 31,  # 树的最大叶子数
    'random_state': 42
}

xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 20,  # 提升树的数量
    'learning_rate': 0.1,  # 学习率
    'max_depth': 8,  # 树的最大深度
    'random_state': 42
}

# 10. 训练模型
rf_model = RandomForestRegressor(**rf_params)
lgb_model = lgb.LGBMRegressor(**lgb_params)
xgb_model = xgb.XGBRegressor(**xgb_params)

# 11. 评估模型
rf_results = train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "随机森林")
lgb_results = train_and_evaluate_model(lgb_model, X_train, X_test, y_train, y_test, "LGBM")
xgb_results = train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")

# 加入交叉验证部分
def cross_validate_model(model, X, y, model_name):
    mse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"\n{model_name}模型 5 折交叉验证结果：")
    print(f"均方误差：{mse_scores.mean()}")
    print(f"平均绝对误差：{mae_scores.mean()}")
    print(f"R方 分数：{r2_scores.mean()}")

    return {
        'mse': mse_scores.mean(),
        'mae': mae_scores.mean(),
        'r2': r2_scores.mean()
    }


rf_cv_results = cross_validate_model(rf_model, X_scaled, y, "随机森林")
lgb_cv_results = cross_validate_model(lgb_model, X_scaled, y, "LGBM")
xgb_cv_results = cross_validate_model(xgb_model, X_scaled, y, "XGBoost")

# 12. 模型比较
models = ['随机森林', 'LGBM', 'XGBoost']
mse_values = [rf_results['mse'], lgb_results['mse'], xgb_results['mse']]
mae_values = [rf_results['mae'], lgb_results['mae'], xgb_results['mae']]
r2_values = [rf_results['r2'], lgb_results['r2'], xgb_results['r2']]

mse_cv_values = [rf_cv_results['mse'], lgb_cv_results['mse'], xgb_cv_results['mse']]
mae_cv_values = [rf_cv_results['mae'], lgb_cv_results['mae'], xgb_cv_results['mae']]
r2_cv_values = [rf_cv_results['r2'], lgb_cv_results['r2'], xgb_cv_results['r2']]

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.bar([i - 0.2 for i in range(len(models))], mse_values, width=0.4, color='skyblue', label='单次评估')
plt.bar([i + 0.2 for i in range(len(models))], mse_cv_values, width=0.4, color='lightblue', label='交叉验证')
plt.title('均方误差 (MSE)')
plt.xticks(range(len(models)), models, rotation=45)
plt.legend()

plt.subplot(1, 3, 2)
plt.bar([i - 0.2 for i in range(len(models))], mae_values, width=0.4, color='lightgreen', label='单次评估')
plt.bar([i + 0.2 for i in range(len(models))], mae_cv_values, width=0.4, color='green', label='交叉验证')
plt.title('平均绝对误差 (MAE)')
plt.xticks(range(len(models)), models, rotation=45)
plt.legend()

plt.subplot(1, 3, 3)
plt.bar([i - 0.2 for i in range(len(models))], r2_values, width=0.4, color='salmon', label='单次评估')
plt.bar([i + 0.2 for i in range(len(models))], r2_cv_values, width=0.4, color='pink', label='交叉验证')
plt.title('R方 分数')
plt.xticks(range(len(models)), models, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# 13. 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_results['predictions'], color='blue', label='随机森林预测值')
plt.scatter(y_test, lgb_results['predictions'], color='green', label='LGBM预测值')
plt.scatter(y_test, xgb_results['predictions'], color='purple', label='XGBoost预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='完美预测线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('模型预测结果比较')
plt.legend()
plt.show()

# 14. 特征重要性评估
def plot_feature_importance(model, features, title):
    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(title)
    plt.show()


plot_feature_importance(rf_model, selected_features, "随机森林特征重要性")
plot_feature_importance(lgb_model, selected_features, "LGBM特征重要性")
plot_feature_importance(xgb_model, selected_features, "XGBoost特征重要性")

# 绘制随机森林第一棵树
dot_data = export_graphviz(rf_model.estimators_[0], out_file=None,
                           feature_names=selected_features,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('/mnt/random_forest_tree', format='png', cleanup=True, view=False)

# 绘制 LightGBM 第一棵树
lgb.create_tree_digraph(lgb_model, tree_index=0, format='png',
                        name='lightgbm_tree',
                        filename='C:\\Users\\86185\\Desktop\\新建文件夹\\lightgbm_tree').render(view=False)

# 绘制 XGBoost 第一棵树
xgb.to_graphviz(xgb_model, tree_idx=0).render('C:\\Users\\86185\\Desktop\\新建文件夹\\xgboost_tree', format='png', cleanup=True, view=False)