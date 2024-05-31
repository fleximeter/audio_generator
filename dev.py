import corpus
import featurizer

ds = corpus.load_audio("data/train")
for file in ds:
    featurizer.featurize(file)

print(ds[0]["power_spectrogram"].shape)
print(ds[0]["audio"].shape)

