import torch

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
except:
    print("Please install torch_geometric to run this code.")


class GeoInterpGCN(torch.nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 1) -> None:
        super().__init__()
        # gcn layers
        self.layer1 = GCNConv(input_dim, input_dim * 2)
        self.layer2 = GCNConv(input_dim * 2, input_dim * 2)
        self.layer3 = GCNConv(input_dim * 2, output_dim)

        # pooling layer
        self.pool = global_mean_pool

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        # dataloader creates this automatically
        batch = data.batch

        x = self.layer1(x, edge_index).relu()
        x = self.layer2(x, edge_index).relu()
        x = self.layer3(x, edge_index)

        x = self.pool(x, batch)

        return x

if __name__ == "__main__":
    # 创建一些示例图
    # 子图1：4个节点，2条边
    x1 = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)  # 节点特征
    edge_index1 = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)  # 边的连接
    edge_index1 = edge_index1.t().contiguous()  # 转换为 COO 格式，t() 转置，确保形状为 [2, num_edges]
    data1 = Data(x=x1, edge_index=edge_index1)

    # 子图2：3个节点，2条边
    x2 = torch.tensor([[1, 1], [0, 0], [1, 0]], dtype=torch.float)
    edge_index2 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_index2 = edge_index2.t().contiguous()  # 转换为 COO 格式
    data2 = Data(x=x2, edge_index=edge_index2)

    # 创建图的列表
    data_list = [data1, data2]

    # 使用 DataLoader 进行批量处理
    loader = DataLoader(data_list, batch_size=2, shuffle=True)

    # 创建模型
    model = GeoInterpGCN(input_dim=2, output_dim=1)

    # 训练过程中的一个示例
    for batch in loader:
        output = model(batch)
        print(output)
