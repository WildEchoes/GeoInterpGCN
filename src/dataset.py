import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

try:
    import torch
    from torch_geometric.data import Data, Dataset
    from torch_geometric.utils import to_undirected
except:
    print("Need to install torch-geometric")
"""
@author yy/袁野
@date 2024/12/2 
@description 图数据集构造
"""


class GeoDataset:
    def __init__(self, datafile: str, label_name: str):
        """
        datadir: input feature
        label_name: name of the label column (e.g., 'Al2O3', 'Fe2O3')
        注意单位：x、y都是km
        """
        assert os.path.isfile(datafile), f"File {datafile} does not exist"

        self.data_list = []

        df = pd.read_csv(datafile)
        df_selected = df[['x_', 'y_', 'dem', label_name]]
        del df

        points = df_selected[['x_', 'y_']].values
        values = df_selected[[label_name]].values

        tree = KDTree(points)
        _, indices = tree.query(points, k=9)

        # 子图边 COO
        e_idx = np.array([[0] * 9, [0, 1, 2, 3, 4, 5, 6, 7, 8]])
        e_idx_undirected = to_undirected(torch.tensor(e_idx, dtype=torch.long))

        for i in range(len(points)):
            sub_grap_data = df_selected.values[indices[i]]

            sub_x = sub_grap_data.copy()
            sub_x[0][3] = 0.0  # 中心点 label值为0
            sub_y = values[i]  # 只保留中心点的y值

            x = torch.tensor(sub_x, dtype=torch.float)
            y = torch.tensor(sub_y, dtype=torch.float).view(-1, 1)

            data_obj = Data(x=x, y=y, edge_index=e_idx_undirected)
            self.data_list.append(data_obj)

    @property
    def get(self):
        return self.data_list


if __name__ == "__main__":
    dataset = GeoDataset("E:/WorkSpace/GeoInterpGCN/data/chem_dem.csv", "Al2O3")
    l = dataset.get
    print(f"dataset shape: {len(l)}*{l[0].x.size()}")
    pass
