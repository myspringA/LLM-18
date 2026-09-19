# 使用TensorFlow实现Boston房价预测（1层隐藏层，16个神经元）
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 获取脚本所在目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))

# 数据加载（从housing.csv读取，无表头，空格分隔）
data = pd.read_csv(os.path.join(script_dir, 'housing.csv'), header=None, sep='\s+')
x = data.iloc[:, :-1].values  # 前13列为特征
y = data.iloc[:, -1].values  # 最后一列为目标
print(f"数据形状: 特征 {x.shape}, 目标 {y.shape}")

# 将y转换形状
y = y.reshape(-1, 1)

# 数据规范化（MinMaxScaler将特征缩放到[0,1]）
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 划分训练集和测试集（75%训练，25%测试）
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
print(f"训练集: {train_x.shape[0]} 条, 测试集: {test_x.shape[0]} 条")

# 构建神经网络：1层隐藏层（16个神经元，ReLU激活）+ 输出层
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(1)
])

# 定义优化器和损失函数
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)

# 查看模型结构
model.summary()

# 训练模型
max_epoch = 500
history = model.fit(train_x, train_y, epochs=max_epoch, verbose=0)

# 绘制Loss曲线
plt.figure(figsize=(10, 5))
plt.plot(np.arange(max_epoch), history.history['loss'])
plt.title('Loss Value in All Iterations')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tf16_loss.png'))
plt.show()

# 测试集预测
predict_list = model.predict(test_x).flatten()
real_values = test_y.flatten()

# 计算评估指标
mse = np.mean((predict_list - real_values) ** 2)
mae = np.mean(np.abs(predict_list - real_values))
print(f"\n测试集评估 - MSE: {mse:.2f}, MAE: {mae:.2f}")

# 绘制真实值与预测值的散点图
plt.figure(figsize=(10, 5))
x_idx = np.arange(len(real_values))
plt.scatter(x_idx, predict_list, c='red', label='Predict', alpha=0.7)
plt.scatter(x_idx, real_values, c='blue', label='Real', alpha=0.7)
plt.legend(loc='best')
plt.title('Prediction vs Real House Prices')
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tf16_predict_vs_real.png'))
plt.show()
