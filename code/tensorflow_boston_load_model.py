# 加载已保存的Boston房价预测模型，进行预测和评估
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# ======================== 加载模型 ========================
model_path = os.path.join(script_dir, 'boston_model.keras')
print(f'正在加载模型: {model_path}')
model = tf.keras.models.load_model(model_path)
print('模型加载成功！')
model.summary()

# ======================== 加载数据 ========================
data = pd.read_csv(os.path.join(script_dir, 'housing.csv'), header=None, sep='\s+')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 数据规范化（必须与训练时使用相同的 MinMaxScaler）
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 划分训练集和测试集（random_state 需与训练时一致）
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)

# ======================== 预测 ========================
predict_list = model.predict(test_x).flatten()
real_values = test_y.flatten()

# 评估
mse = np.mean((predict_list - real_values) ** 2)
mae = np.mean(np.abs(predict_list - real_values))
print(f'\n测试集评估 - MSE: {mse:.2f}, MAE: {mae:.2f}')

# ======================== 可视化 ========================
# 预测 vs 真实值散点图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_idx = np.arange(len(real_values))
plt.scatter(x_idx, predict_list, c='red', label='Predict', alpha=0.7)
plt.scatter(x_idx, real_values, c='blue', label='Real', alpha=0.7)
plt.legend(loc='best')
plt.title('Prediction vs Real (Loaded Model)')
plt.xlabel('Sample Index')
plt.ylabel('House Price')

# 预测值 vs 真实值 拟合图
plt.subplot(1, 2, 2)
plt.scatter(real_values, predict_list, alpha=0.6, edgecolors='k')
plt.plot([real_values.min(), real_values.max()],
         [real_values.min(), real_values.max()], 'r--', label='Perfect Fit')
plt.legend(loc='best')
plt.title('Predicted vs Actual')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'loaded_model_result.png'))
plt.show()

# ======================== 单条预测示例 ========================
print('\n--- 单条预测示例（取测试集前5条） ---')
for i in range(5):
    print(f'  真实值: {real_values[i]:.1f}  预测值: {predict_list[i]:.1f}  误差: {abs(real_values[i] - predict_list[i]):.1f}')
