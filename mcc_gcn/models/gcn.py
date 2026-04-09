import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNNet(nn.Module):
    """Graph Convolutional Network for multi-component crystal classification.

    Args:
        num_classes: Number of output classes (4 for multi-class, 2 for binary).
        is_large: Use larger hidden dimensions (v1/v2 config) if True.
    """

    CONFIGS = {
        False: dict(hidden1=128, hidden2=64, hidden3=64, readout=64, dense1=64, dense2=32),
        True:  dict(hidden1=256, hidden2=256, hidden3=128, readout=128, dense1=128, dense2=64),
    }

    def __init__(self, num_classes=4, is_large=False):
        super().__init__()
        cfg = self.CONFIGS[is_large]
        self.input_dim = 34
        self.output_dim = num_classes
        self.dropout_rate = 0.208

        self.conv1 = GCNConv(self.input_dim, cfg['hidden1'])
        self.conv2 = GCNConv(cfg['hidden1'], cfg['hidden2'])
        self.conv3 = GCNConv(cfg['hidden2'], cfg['hidden3'])

        self.bn1 = nn.BatchNorm1d(cfg['hidden1'])
        self.bn2 = nn.BatchNorm1d(cfg['hidden2'])
        self.bn3 = nn.BatchNorm1d(cfg['hidden3'])

        self.fc1 = nn.Linear(cfg['readout'], cfg['dense1'])
        self.bn4 = nn.BatchNorm1d(cfg['dense1'])
        self.fc2 = nn.Linear(cfg['dense1'], cfg['dense2'])
        self.bn5 = nn.BatchNorm1d(cfg['dense2'])
        self.fc_out = nn.Linear(cfg['dense2'], self.output_dim)

    def ft_setting(self, train_dense_layer=1):
        """Freeze parameters for fine-tuning, only training the last N dense layers."""
        for param in self.parameters():
            param.requires_grad = False

        if train_dense_layer == 1:
            for param in self.fc_out.parameters():
                param.requires_grad = True
        elif train_dense_layer == 2:
            for layer in [self.fc2, self.fc_out, self.bn5]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif train_dense_layer == 3:
            for layer in [self.fc1, self.fc2, self.fc_out, self.bn4, self.bn5]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.bn4(self.fc1(x))), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.bn5(self.fc2(x))), p=self.dropout_rate, training=self.training)
        return self.fc_out(x)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0, 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

        bs = batch.y.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    return all_labels, all_preds, total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)

            bs = batch.y.size(0)
            total_samples += bs
            total_loss += loss.item() * bs
            all_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    return all_labels, all_preds, total_loss / total_samples
