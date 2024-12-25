import featurizer
import dataset
import torch

ds = dataset.AudioDataset("data/train", 10, 1024)
dl = torch.utils.data.DataLoader(ds, 10, True)
i = 0
for X, y in dl:
    if i == 0:
        print(type(X))
        print(X.shape)
    i += 1
