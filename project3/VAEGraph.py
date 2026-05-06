import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.data import Dataset, DataLoader
import numpy as np
import networkx as nx
from typing import List, Tuple
import math
import argparse

from data import GraphDataset
from data import GC


def collate_graphs(batch: List[Tuple[np.ndarray, int]]):
    # batch: list of (adjacency numpy array, N, node_features, node_features)
    Ns = [a.shape[0] for a, _, _ in batch]
    max_N = max(Ns)
    batch_size = len(batch)
    adj_tensor = torch.zeros((batch_size, max_N, max_N), dtype=torch.float32)
    features_tensor = torch.zeros((batch_size, max_N, 7), dtype=torch.float32)  
    mask = torch.zeros((batch_size, max_N), dtype=torch.float32)
    for i, (adj, _, node_feat) in enumerate(batch):
        n = adj.shape[0]
        adj_tensor[i, :n, :n] = torch.from_numpy(adj)
        mask[i, :n] = 1.0
        features_tensor[i, :n] = torch.from_numpy(np.array(node_feat, dtype=np.float32))
    return adj_tensor, mask, features_tensor


class GraphTorchDataset(Dataset):
    def __init__(self, graph_dataset: GraphDataset):
        self.graph_dataset = graph_dataset

    def __len__(self):
        return len(self.graph_dataset.dataset)

    def __getitem__(self, idx):
        G, raw_G = self.graph_dataset[idx]
        adj = nx.to_numpy_array(G, dtype=float)
        # This is list[list] of node features, features are 7 long.
        return adj, G.number_of_nodes(), raw_G["node_feat"]

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, adj, h, mask):
        # adj: (B,N,N), h: (B,N,F), mask: (B,N)
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(-1)  # (B,N)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-12), -0.5)  # (B,N)
        # normalize adjacency: (B,N,1) * (B,N,N) * (B,1,N) -> (B,N,N)
        neigh_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        neigh = torch.matmul(neigh_adj, h)
        out = self.lin(neigh)
        out = F.relu(out)
        out = out * mask.unsqueeze(-1)
        return out


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class NodeVAE(nn.Module):
    def __init__(self, node_feat_dim=7, hidden_dim=64, z_dim=16, n_gcn_layers=2, beta=0.1):
        super().__init__()
        self.beta = beta
        self.z_dim = z_dim

        # Encoder
        layers = []
        in_dim = node_feat_dim
        for _ in range(n_gcn_layers):
            layers.append(GCNLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.gcn_layers = nn.ModuleList(layers)
        self.to_mu = nn.Linear(hidden_dim, z_dim)
        self.to_logvar = nn.Linear(hidden_dim, z_dim)
        self.edge_bias = nn.Parameter(torch.tensor(-2.0))

        # Decoder
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, adj, mask, features):
        # B, N, _ = adj.shape
        # features: (B, N, 7) - node features
        h = features
        for gcn in self.gcn_layers:
            h = gcn(adj, h, mask)
        mu = self.to_mu(h) * mask.unsqueeze(-1)
        logvar = self.to_logvar(h) * mask.unsqueeze(-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z, mask):
        # z: (B,N,z_dim)
        # produce symmetric logits matrix (B,N,N)
        scores = torch.matmul(z, z.transpose(-1, -2)) / math.sqrt(self.z_dim)
        scores = 0.5 * (scores + scores.transpose(-1, -2))
        scores += self.edge_bias
        # prevent self-loops: set diagonal to large negative
        diag_mask = torch.eye(scores.size(-1), device=scores.device).unsqueeze(0)
        scores = scores - 1e6 * diag_mask
        # mask out padded rows/cols
        inv_mask = (1.0 - mask).unsqueeze(-1)
        scores = scores - 1e6 * (inv_mask + inv_mask.transpose(-1, -2))
        return scores

    def forward(self, adj, target_adj, mask, features):
        # adj: (B,N,N) binary, target_adj: (B,N,N) binary, mask: (B,N), features: (B,N,7)
        mu, logvar = self.encode(adj, mask, features)
        z = self.reparameterize(mu, logvar)

        logits = self.decode_logits(z, mask)

        # Reconstruction log-prob
        bern = td.Independent(td.Bernoulli(logits=logits), 2)
        recon = bern.log_prob(target_adj)

        # KL per node
        var = torch.exp(logvar)
        kl_node = 0.5 * (mu.pow(2) + var - 1.0 - logvar)
        kl_sum = (kl_node.sum(-1) * mask).sum(-1)

        # recon is a scalar per graph (sums over N,N)
        elbo = recon - self.beta * kl_sum
        # average over batch
        # MAYBE NOT DO THIS WITH NUMBER OF NODES?
        loss = -(elbo / mask.sum(-1)).mean()
        return loss, {'elbo': elbo.mean().item(), 'recon': recon.mean().item(), 'kl': kl_sum.mean().item()}

    def sample(self, num_nodes: int, device='cpu'):
        # sample z ~ N(0,1) per node
        z = torch.randn((1, num_nodes, self.z_dim), device=device)
        logits = self.decode_logits(z, torch.ones((1, num_nodes), device=device))
        probs = torch.sigmoid(logits)
        adj = (torch.rand_like(probs) < probs).float()
        return adj.squeeze(0)


def train_loop(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):

        current_beta = min(0.001, 0.001 * (epoch / 50.0))
        model.beta = current_beta

        total_loss = 0.0
        for adj, mask, features in dataloader:
            hat_adj = adj.to(device) + torch.eye(adj.size(-1), device=device).unsqueeze(0)  # add self-loops for encoding
            mask = mask.to(device)
            features = features.to(device)
            optimizer.zero_grad()
            loss, stats = model(hat_adj, adj.to(device), mask, features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * adj.size(0)
        avg = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg:0.6f}  elbo: {stats['elbo']:0.6f}  kl: {stats['kl']:0.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--z-dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save', type=str, default='project3/vae_graph.pt')
    parser.add_argument('--num-samples', type=int, default=1, help='how many graphs to sample in `sample` mode')
    args = parser.parse_args()

    device = torch.device(args.device)

    ds = GraphTorchDataset(GC)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)

    model = NodeVAE(node_feat_dim=7, hidden_dim=64, z_dim=args.z_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loop(model, loader, opt, device, epochs=args.epochs)
    torch.save(model.state_dict(), args.save)
    print('Saved model to', args.save)


if __name__ == '__main__':
    main()
