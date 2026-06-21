import torch
import numpy as np
from node_features import GCN, count_parameters, pca_the_features, create_dataset

def test_gcn_model():
    model = GCN(input_size=10, num_classes=10)
    assert count_parameters(model) > 0
    print("test_gcn_model passed")

def test_pca_the_features():
    features = np.random.rand(10000, 50)
    train_pca, val_pca = pca_the_features(10, features)
    assert train_pca.shape == (9200, 10)
    assert val_pca.shape == (800, 10)
    print("test_pca_the_features passed")

if __name__ == "__main__":
    test_gcn_model()
    test_pca_the_features()
