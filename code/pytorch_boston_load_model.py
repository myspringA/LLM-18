# 加载已保存的PyTorch Boston房价预测模型，进行预测和评估
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# ======================== 定义网络结构（需与训练时一致） ========================
class BostonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# ======================== 加载模型 ========================
model = BostonNet()
model_path = os.path.join(script_dir, 'boston_model_pytorch.pth')
print(f'正在加载模型: {model_path}')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print('模型加载成功！')
print(model)

# ======================== 加载数据 ========================
data = pd.read_csv(os.path.join(script_dir, 'housing.csv'), header=None, sep='\s+')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 归一化（与训练时一致）
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 划分（random_state 需与训练时一致）
_, test_x, _, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# ======================== 预测 ========================
with torch.no_grad():
    test_pred = model(test_x).numpy().flatten()
    test_real = test_y.numpy().flatten()

# 评估
mse = np.mean((test_pred - test_real) ** 2)
mae = np.mean(np.abs(test_pred - test_real))
print(f'\n测试集评估 - MSE: {mse:.2f}, MAE: {mae:.2f}')

# ======================== 可视化 ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 预测 vs 真实值散点图
x_idx = np.arange(len(test_real))
axes[0].scatter(x_idx, test_pred, c='red', label='Predict', alpha=0.7)
axes[0].scatter(x_idx, test_real, c='blue', label='Real', alpha=0.7)
axes[0].legend(loc='best')
axes[0].set_title('Prediction vs Real (Loaded Model)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('House Price')
axes[0].grid(True)

# 预测值 vs 真实值拟合图
axes[1].scatter(test_real, test_pred, alpha=0.6, edgecolors='k')
axes[1].plot([test_real.min(), test_real.max()],
             [test_real.min(), test_real.max()], 'r--', label='Perfect Fit')
axes[1].legend(loc='best')
axes[1].set_title('Predicted vs Actual')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'pytorch_loaded_result.png'))
plt.show()

# ======================== 单条预测示例 ========================
print('\n--- 前5条预测 ---')
for i in range(5):
    print(f'  真实值: {test_real[i]:.1f}  预测值: {test_pred[i]:.1f}  误差: {abs(test_real[i] - test_pred[i]):.1f}')
