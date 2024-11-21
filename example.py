import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 假设我们有一些预处理好的数据
# 节点特征 X, 邻接矩阵 edge_index, 目标值 y
# 特征维度为 [num_nodes, feature_dim]，这里 feature_dim = 2 (经度, 纬度)
# 浓度值存储在另一个张量中
X = torch.tensor([
    [0.5, 0.5],  # 经度, 纬度 (点A)
    [0.6, 0.6],  # 经度, 纬度 (点B)
    [0.7, 0.7],  # 经度, 纬度 (点C)
    [0.8, 0.8]   # 经度, 纬度 (点D)
], dtype=torch.float)

edge_index = torch.tensor([[0, 1, 0, 2, 0, 3],
                           [1, 0, 2, 0, 3, 0]], dtype=torch.long)

concentration_values = torch.tensor([0.0, 0.65, 0.75, 0.85], dtype=torch.float)

y = torch.tensor([0.55], dtype=torch.float)  # 点A的真实浓度值

# 创建Data对象
data = Data(x=X, edge_index=edge_index, concentration=concentration_values, y=y)

class GCNPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)  # 输入是隐藏层特征和浓度特征

    def forward(self, data):
        x, edge_index, concentration = data.x, data.edge_index, data.concentration
        
        # 图卷积操作
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # 获取点A的特征
        target_node_idx = 0  # 假设点A是第一个节点
        target_features = x[target_node_idx]
        
        # 获取相邻节点的浓度特征
        neighbor_indices = (edge_index[1] == target_node_idx).nonzero().squeeze()
        neighbor_concentrations = concentration[edge_index[0][neighbor_indices]]
        neighbor_features = x[edge_index[0][neighbor_indices]].mean(dim=0)
        
        # 合并特征
        combined_features = torch.cat((target_features, neighbor_features))
        
        # 全连接层进行最终预测
        out = self.fc(combined_features)
        return out.view(-1)

# 定义模型参数
input_dim = 2  # 经度, 纬度
hidden_dim = 8
output_dim = 1

model = GCNPredictor(input_dim, hidden_dim, output_dim)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 将所有数据视为训练数据（仅用于演示）
train_mask = torch.ones(1, dtype=torch.bool)  # 只训练点A的目标值

# 训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

# 测试模型
model.eval()
with torch.no_grad():
    pred = model(data)
    print(f'Predicted value for node A: {pred.item()}')
    print(f'True value for node A: {data.y.item()}')






