from torch.utils.data import DataLoader
from utils.dataloader import MRIDataset

dataset = MRIDataset("data/processed/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for x, y in loader:
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    break