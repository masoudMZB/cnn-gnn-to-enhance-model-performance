# cnn-gnn-to-enhance-model-performance

to download features and use it in your own code:
- features extracted using efficient net : https://drive.google.com/file/d/1YgDCP82zGNdwnY3N1jG4jujKZm5It8r9/view?usp=drive_link
- features extracted using squeeze net 512_13_13 shape : https://drive.google.com/file/d/1KLX2hEeuRTSYX_eSPqRmNl8iqQf0_uRC/view?usp=drive_link 
- features extracted using squeeze net 86528 shape : https://drive.google.com/file/d/1-00pBLYX4E9V4gx0eneVgRh_ZfVDDDA4/view?usp=drive_link

# what are the files : 
Draw plots jupter notebook use saved results to plot some useful information.
<br>
node_features jupyter notebook is used to train models.


# How to generate Datasets.
Mnist dataset is prepared by the Torch-geometric which is available with this code
> from torch_geometric.datasets import MNISTSuperpixels

# CIFAR-10 Graph Dataset Preparation

This project demonstrates how to prepare the CIFAR-10 dataset for graph-based deep learning tasks using PyTorch and PyTorch Geometric.

## Overview

The code provided sets up a pipeline to transform the CIFAR-10 image dataset into a graph-structured dataset. This transformation involves segmenting the images using the SLIC algorithm and creating a k-nearest neighbors graph from the resulting superpixels.

## Dependencies

- PyTorch
- torchvision
- PyTorch Geometric

## Code Explanation

```python
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch_geometric.transforms import ToSLIC, KNNGraph

# Define the transform pipeline
transform = T.Compose([
    T.ToTensor(),  # Convert image to PyTorch tensor
    ToSLIC(n_segments=117, add_img=False),  # Apply ToSLIC transform
    KNNGraph(k=15)
])

# Load the CIFAR-10 dataset with the defined transform
dataset_train = CIFAR10(root='/tmp/CIFAR10', train=True, download=True, transform=transform)
dataset_test = CIFAR10(root='/tmp/CIFAR10', train=False, download=True, transform=transform)
```

### Transform Pipeline

1. `T.ToTensor()`: Converts the input image to a PyTorch tensor.
2. `ToSLIC(n_segments=117, add_img=False)`: Applies the SLIC (Simple Linear Iterative Clustering) algorithm to segment the image into 117 superpixels. The `add_img=False` parameter means the original image data won't be included in the output.
3. `KNNGraph(k=15)`: Constructs a k-nearest neighbors graph from the superpixels, with each node connected to its 15 nearest neighbors.

### Dataset Loading

The code loads both the training and testing sets of CIFAR-10, applying the defined transform to each image. The datasets are downloaded to the '/tmp/CIFAR10' directory.

## Usage

This code prepares the CIFAR-10 dataset for graph-based deep learning tasks. After running this code, you can use the `dataset_train` and `dataset_test` objects to train and evaluate graph neural networks on the CIFAR-10 dataset.

## Customization

You can adjust the following parameters to experiment with different graph structures:

- `n_segments` in `ToSLIC()`: Controls the number of superpixels.
- `k` in `KNNGraph()`: Determines the number of nearest neighbors for each node.

## Note

Ensure you have sufficient disk space and memory to handle the transformed dataset, as graph representations can be memory-intensive.
