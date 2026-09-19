# 使用PyTorch实现Boston房价预测（1层隐藏层，16个神经元）
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# ======================== 数据加载与预处理 ========================
data = pd.read_csv(os.path.join(script_dir, 'housing.csv'), header=None, sep='\s+')
x = data.iloc[:, :-1].values  # 前13列为特征
y = data.iloc[:, -1].values   # 最后一列为目标
y = y.reshape(-1, 1)

# 归一化
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
print(f"训练集: {train_x.shape[0]} 条, 测试集: {test_x.shape[0]} 条")

# 转为PyTorch张量
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x  = torch.tensor(test_x, dtype=torch.float32)
test_y  = torch.tensor(test_y, dtype=torch.float32)

# ======================== 定义神经网络 ========================
class BostonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 16),   # 输入层: 13特征 → 隐藏层: 16神经元
            nn.ReLU(),           # 激活函数
            nn.Linear(16, 1)     # 隐藏层: 16神经元 → 输出层: 1（房价）
        )

    def forward(self, x):
        return self.net(x)

model = BostonNet()
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# ======================== 训练配置 ========================
criterion = nn.MSELoss()                    # 损失函数: 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 优化器: Adam
max_epoch = 500

# ======================== 训练 ========================
loss_history = []
for epoch in range(max_epoch):
    # 前向传播
    y_pred = model(train_x)
    loss = criterion(y_pred, train_y)

    # 反向传播 + 更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {loss.item():.4f}")

# ======================== 保存模型 ========================
model_path = os.path.join(script_dir, 'boston_model_pytorch.pth')
torch.save(model.state_dict(), model_path)
print(f"\n模型已保存到: {model_path}")

# ======================== 评估 ========================
model.eval()
with torch.no_grad():
    test_pred = model(test_x).numpy().flatten()
    test_real = test_y.numpy().flatten()
    mse = np.mean((test_pred - test_real) ** 2)
    mae = np.mean(np.abs(test_pred - test_real))
    print(f"测试集评估 - MSE: {mse:.2f}, MAE: {mae:.2f}")

# ======================== 可视化 ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss曲线
axes[0].plot(loss_history)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].grid(True)

# 预测 vs 真实值
x_idx = np.arange(len(test_real))
axes[1].scatter(x_idx, test_pred, c='red', label='Predict', alpha=0.7)
axes[1].scatter(x_idx, test_real, c='blue', label='Real', alpha=0.7)
axes[1].legend(loc='best')
axes[1].set_title('Prediction vs Real')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('House Price')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'pytorch_result.png'))
plt.show()

# 打印前5条预测
print('\n--- 前5条预测 ---')
for i in range(5):
    print(f"  真实值: {test_real[i]:.1f}  预测值: {test_pred[i]:.1f}  误差: {abs(test_real[i] - test_pred[i]):.1f}")
