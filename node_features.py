import os
import torch
import torchvision.models as models
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
import statistics
import pickle
import numpy as np

embedding_size = 10

class GCN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(GCN, self).__init__()
        self.initial_conv = GCNConv(input_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.out = Linear(embedding_size*2, num_classes)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out, hidden

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_to_draw_metrics(data_loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_total = 0
    result_data = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            test_loss = loss_fn(pred, batch.y)
            total_loss += test_loss.item()

            _, test_predicted = torch.max(pred.data, 1)
            total_total += batch.y.size(0)
            total_correct += (test_predicted == batch.y).sum().item()

            for i in range(len(batch.y)):
                result_data.append([batch.y[i].item(), test_predicted[i].item()])

    accuracy = total_correct / total_total
    avg_loss = total_loss / len(data_loader)

    columns = ["Real Label", "Predicted Label"]
    df = pd.DataFrame(result_data, columns=columns)

    cm = confusion_matrix(df['Real Label'], df['Predicted Label'])
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(df['Real Label'], df['Predicted Label']))

    return avg_loss, accuracy, df

def pca_the_features(dim, features):
    train_indices = range(9200)
    val_indices = range(9200, 10000)

    train_features = features[:9200]
    val_features = features[9200:]

    pca = PCA(n_components=dim)
    pca.fit(train_features)
    train_features_pca = pca.transform(train_features)
    val_features_pca = pca.transform(val_features)

    return train_features_pca, val_features_pca

def create_dataset(train_features_pca, val_features_pca):
    dataset = MNISTSuperpixels(root='/tmp', train=False)

    train_indices = range(9200)
    val_indices = range(9200, 10000)

    train_dataset = []
    val_dataset = []

    for i, graph in enumerate(dataset):
        num_nodes = graph.num_nodes
        if i in train_indices:
            graph.x = torch.tensor(train_features_pca[i], dtype=torch.float).unsqueeze(0).repeat(num_nodes, 1)
            train_dataset.append(graph)
        elif i in val_indices:
            graph.x = torch.tensor(val_features_pca[i - 9200], dtype=torch.float).unsqueeze(0).repeat(num_nodes, 1)
            val_dataset.append(graph)

    return train_dataset, val_dataset

def train_model(dim, train_dataset, val_dataset,num_classes=10, draw_metrics=False):
    import warnings
    warnings.filterwarnings("ignore")

    def train_and_test(data_loader, model, optimizer, loss_fn, device):
        model.train()
        correct = 0
        total = 0
        total_loss = 0

        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            total_loss += loss.item()

        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy

    def evaluate(data_loader, model, loss_fn, device):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_total = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
                test_loss = loss_fn(pred, batch.y)
                total_loss += test_loss.item()

                _, test_predicted = torch.max(pred.data, 1)
                total_total += batch.y.size(0)
                total_correct += (test_predicted == batch.y).sum().item()

        accuracy = total_correct / total_total
        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy

    print("Starting training...")

    best_accuracy = [0.0] * 5
    best_accuracy_train = [0.0] * 5
    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    patience = 5

    for run_time in range(5):
        model = GCN(dim, num_classes)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        NUM_GRAPHS_PER_BATCH = 64
        train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
        test_loader = DataLoader(val_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

        early_stop_counter = 0

        for epoch in range(40):
            train_loss, train_accuracy = train_and_test(train_loader, model, optimizer, loss_fn, device)
            test_loss, test_accuracy = evaluate(test_loader, model, loss_fn, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch} | Train Loss {train_loss:.4f} | Train Accuracy {train_accuracy:.4f} | Test Loss {test_loss:.4f} | Test Accuracy {test_accuracy:.4f}")

            if train_accuracy > best_accuracy_train[run_time]:
                best_accuracy_train[run_time] = train_accuracy

            if test_accuracy > best_accuracy[run_time]:
                best_accuracy[run_time] = test_accuracy
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Validation accuracy did not improve for several epochs. Stopping early.")
                    break

        all_train_losses.append(train_losses)
        all_train_accuracies.append(train_accuracies)
        all_test_losses.append(test_losses)
        all_test_accuracies.append(test_accuracies)

    print(f'best accuraces for test was : {best_accuracy}')
    if len(best_accuracy) > 1:
        print(f'stddev for test was  : {statistics.stdev(best_accuracy)}')

    print(f'best accuraces for train was : {best_accuracy_train}')
    if len(best_accuracy_train) > 1:
        print(f'stddev for train was  : {statistics.stdev(best_accuracy_train)}')

    if draw_metrics:
        evaluate_to_draw_metrics(test_loader, model, loss_fn, device)

    return all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies

def run_experiment(dim, features, show_metric=False):
    train_features_pca, val_features_pca = pca_the_features(dim, features)
    train_dataset, val_dataset = create_dataset(train_features_pca, val_features_pca)
    logs = train_model(dim, train_dataset, val_dataset, num_classes=10, draw_metrics=show_metric)
    return logs

def run_experiment_no_pca(dim, train_features, val_features, show_metric=False):
    train_dataset, val_dataset = create_dataset(train_features, val_features)
    logs = train_model(dim, train_dataset, val_dataset, num_classes=10, draw_metrics=show_metric)
    return logs

if __name__ == "__main__":
    # Example usage - in actual practice, these features would be loaded
    # features = np.load('path_to_features.npy')
    # logs = run_experiment(32, features)
    print("Models defined successfully.")
