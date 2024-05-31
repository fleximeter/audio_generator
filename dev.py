import corpus
import featurizer

ds = corpus.load_audio("data/train")
for file in ds:
    featurizer.featurize(file)

