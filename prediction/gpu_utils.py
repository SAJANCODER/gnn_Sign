import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, inputs):
        X, A = inputs
        # Symmetric Normalization logic
        D_hat = torch.sum(A, dim=-1, keepdim=True)
        D_hat_inv_sqrt = torch.pow(D_hat + 1e-10, -0.5)
        D_hat_inv_sqrt = torch.where(torch.isinf(D_hat_inv_sqrt), torch.zeros_like(D_hat_inv_sqrt), D_hat_inv_sqrt)
        A_norm = D_hat_inv_sqrt * A * D_hat_inv_sqrt.transpose(1, 2)

        X_transformed = self.linear(X)
        output = torch.matmul(A_norm, X_transformed) + self.bias
        return output


class GlobalSumPool(nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)


class GNNModel(nn.Module):
    def __init__(self, num_classes=12):
        super(GNNModel, self).__init__()
        # GCN Block 1
        self.gcn1 = CustomGCNConv(3, 256)
        self.bn1 = nn.BatchNorm1d(256)

        # GCN Block 2
        self.gcn2 = CustomGCNConv(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # GCN Block 3
        self.gcn3 = CustomGCNConv(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = GlobalSumPool()

        # Dense Head
        self.fc1 = nn.Linear(128, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        # GCN Layers (using BatchNorm1d requires permuting for [Batch, Channels, Nodes])
        x = self.gcn1([x, adj])
        x = F.relu(self.bn1(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)

        x = self.gcn2([x, adj])
        x = F.relu(self.bn2(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)

        x = self.gcn3([x, adj])
        x = F.relu(self.bn3(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)

        x = self.pool(x)

        # Dense Layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)

        return self.fc3(x)