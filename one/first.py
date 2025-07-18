import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb

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

# 特征编码与标准化
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['int64', 'float64'])),
                         columns=df.select_dtypes(include=['int64', 'float64']).columns)

# 特征和目标分离
X = df_scaled.drop('popularity', axis=1)
y = df['popularity']

# 特征选择 1：SelectKBest
selector = SelectKBest(f_regression, k=10)
X_selected_kbest = selector.fit_transform(X, y)
selected_features_kbest = X.columns[selector.get_support()]

# 特征选择 2：Recursive Feature Elimination (RFE)
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected_rfe = rfe.fit_transform(X, y)
selected_features_rfe = X.columns[rfe.support_]

# 输出被选择的特征
print(f"通过 SelectKBest 选择的特征：{selected_features_kbest}")
print(f"通过 RFE 选择的特征：{selected_features_rfe}")

# 准备LGBM模型数据
X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(X_selected_kbest, y, test_size=0.2, random_state=42)

# LGBM 超参数调优
lgb_model = lgb.LGBMRegressor()
param_grid_lgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 5, 10],
    'n_estimators': [50, 100, 200]
}
grid_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_lgb.fit(X_train_lgb, y_train_lgb)

print("LGBM最佳参数：", grid_lgb.best_params_)
best_lgb_model = grid_lgb.best_estimator_

# 预测LGBM模型
y_pred_lgb = best_lgb_model.predict(X_test_lgb)

# LGBM模型评估
mse_lgb = mean_squared_error(y_test_lgb, y_pred_lgb)
r2_lgb = r2_score(y_test_lgb, y_pred_lgb)
print(f"LGBM - 均方误差(MSE): {mse_lgb}")
print(f"LGBM - R平方值: {r2_lgb}")

# 交叉验证：LGBM模型
lgb_cv_score = cross_val_score(best_lgb_model, X_selected_kbest, y, cv=5, scoring='neg_mean_squared_error')
print(f"LGBM - 交叉验证均方误差(MSE): {-lgb_cv_score.mean()}")

# 准备XGBoost模型数据
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_selected_rfe, y, test_size=0.2, random_state=42)

# XGBoost 超参数调优
xgb_model = xgb.XGBRegressor()
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0]
}
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train_xgb, y_train_xgb)

print("XGBoost最佳参数：", grid_xgb.best_params_)
best_xgb_model = grid_xgb.best_estimator_

# 预测XGBoost模型
y_pred_xgb = best_xgb_model.predict(X_test_xgb)

# XGBoost模型评估
mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
r2_xgb = r2_score(y_test_xgb, y_pred_xgb)
print(f"XGBoost - 均方误差(MSE): {mse_xgb}")
print(f"XGBoost - R平方值: {r2_xgb}")

# 交叉验证：XGBoost模型
xgb_cv_score = cross_val_score(best_xgb_model, X_selected_rfe, y, cv=5, scoring='neg_mean_squared_error')
print(f"XGBoost - 交叉验证均方误差(MSE): {-xgb_cv_score.mean()}")

# 提供正确和错误样本分析
correct_samples_lgb = [(y_test_lgb.iloc[i], y_pred_lgb[i]) for i in range(len(y_test_lgb)) if abs(y_test_lgb.iloc[i] - y_pred_lgb[i]) < 0.1]
incorrect_samples_lgb = [(y_test_lgb.iloc[i], y_pred_lgb[i]) for i in range(len(y_test_lgb)) if abs(y_test_lgb.iloc[i] - y_pred_lgb[i]) >= 0.1]

correct_samples_xgb = [(y_test_xgb.iloc[i], y_pred_xgb[i]) for i in range(len(y_test_xgb)) if abs(y_test_xgb.iloc[i] - y_pred_xgb[i]) < 0.1]
incorrect_samples_xgb = [(y_test_xgb.iloc[i], y_pred_xgb[i]) for i in range(len(y_test_xgb)) if abs(y_test_xgb.iloc[i] - y_pred_xgb[i]) >= 0.1]

# 显示一些正确样本和错误样本
print("LGBM 正确样本：", correct_samples_lgb[:2])
print("LGBM 错误样本：", incorrect_samples_lgb[:2])
print("XGBoost 正确样本：", correct_samples_xgb[:2])
print("XGBoost 错误样本：", incorrect_samples_xgb[:2])
