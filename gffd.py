import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, Lambda, concatenate
from tensorflow.keras import backend as K

# ==================== 数据预处理 ====================
# 加载数据
df = pd.read_excel('data2019.xlsx', parse_dates=['时间'], index_col='时间')


# 特征工程
def feature_engineering(df):
    # 昼夜标记（夜间辐射为0时标记为0）
    df['is_day'] = df['总辐射(W/m2)'].apply(lambda x: 1 if x > 0 else 0)

    # 滞后特征（前3小时数据）
    for lag in [12, 24, 36]:  # 15分钟间隔 -> 3小时=12步
        df[f'总辐射_lag{lag}'] = df['总辐射(W/m2)'].shift(lag)

    # 滑动窗口统计
    df['temp_rolling_mean'] = df['组件温度(℃)'].rolling(24).mean()  # 6小时平均温度

    return df.dropna()


df = feature_engineering(df)

# 选择特征列
features = ['组件温度(℃)', '温度(°)', '气压(hPa)', '湿度(%)',
            '总辐射(W/m2)', '直射辐射(W/m2)', '散射辐射(W/m2)',
            'is_day', '总辐射_lag12', '总辐射_lag24', 'temp_rolling_mean']
target = '实际发电功率(mw)'

# 归一化
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[[target]])


# 创建时间序列样本
def create_dataset(X, y, time_steps=24, predict_steps=4):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - predict_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps: i + time_steps + predict_steps])  # 预测未来1小时（4个15分钟）
    return np.array(Xs), np.array(ys)


time_steps = 24 * 4  # 使用24小时历史数据（4*24=96步）
predict_steps = 4  # 预测未来1小时
X_seq, y_seq = create_dataset(X, y, time_steps, predict_steps)

# 训练集/测试集划分
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


# ==================== 模型构建 ====================
def attention_layer(inputs, neurons):
    # Bahdanau注意力机制
    x = LSTM(neurons, return_sequences=True)(inputs)
    attention = Dense(1, activation='tanh')(x)
    attention = Lambda(lambda x: K.softmax(x, axis=1))(attention)
    context = Multiply()([x, attention])
    return context


input_layer = Input(shape=(time_steps, len(features)))
lstm_out = LSTM(64, return_sequences=True)(input_layer)
attention_out = attention_layer(lstm_out, 32)
context_vector = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
output = Dense(predict_steps)(context_vector)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')

# ==================== 模型训练 ====================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ==================== 模型评估 ====================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((scaler_y.inverse_transform(y_true) - scaler_y.inverse_transform(y_pred)) ** 2))


y_pred = model.predict(X_test)
print(f"Test RMSE: {rmse(y_test, y_pred):.2f} MW")

# 可视化预测结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(scaler_y.inverse_transform(y_test[100]), label='真实值')
plt.plot(scaler_y.inverse_transform(y_pred[100]), label='预测值')
plt.title('光伏功率预测（LSTM-Attention）')
plt.legend()
plt.show()