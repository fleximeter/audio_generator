import corpus
import featurizer
import torch

ds = corpus.load_audio("data/train")
ds_list = []
for file in ds:
    featurizer.featurize(file)
    for key, val in file.items():
        if type(val) == torch.Tensor:
            ds_list.append(val)

scaler = featurizer.RobustScaler(*featurizer.prepare_robust_scaler(ds_list))