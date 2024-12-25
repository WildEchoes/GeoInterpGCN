import dataset
from src.model import GeoInterpGCN
from torch_geometric.data import  DataLoader

def main():
    model = GeoInterpGCN(input_dim=4, output_dim=1)
    datasets = dataset.GeoDataset("E:/WorkSpace/GeoInterpGCN/data/chem_dem.csv", "Al2O3")
    loader = DataLoader(datasets.get, batch_size=10, shuffle=True)
    for batch in loader:
        output = model(batch)
        print(output)

if __name__ == '__main__':
    main()
