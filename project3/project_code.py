"""
Simple GraphVAE: degree as only extra node feature, hungarian ordering,
balanced BCE, connectivity penalty, cyclical KL.
"""
import os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx, matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import warnings; warnings.filterwarnings('ignore')

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

os.makedirs('simple_model_output', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

EPOCHS      = 200
Z_DIM       = 32
H_DIM       = 128
LR          = 3e-3
BETA_MAX    = 0.01
CONN_LAMBDA = 0.5
N_SAMPLE    = 1000

# ---------------------------------------------------------------
# DATA — atom one-hot (7) + degree (1) = 8 features
# ---------------------------------------------------------------
def add_degree(A, X):
    deg = A.sum(1, keepdim=True)                     # (N, 1)
    lo, hi = deg.min(), deg.max()
    deg_norm = (deg - lo) / (hi - lo + 1e-8)
    return torch.cat([X, deg_norm], dim=1)           # (N, 8)

ds = TUDataset(root='./data', name='MUTAG')
raw_graphs = []
for d in ds:
    N = d.num_nodes
    A = to_dense_adj(d.edge_index, max_num_nodes=N).squeeze(0)
    X = d.x.float()
    raw_graphs.append((A, add_degree(A, X), N))

train_nx = [nx.from_numpy_array(A.numpy()) for A,_,_ in raw_graphs]
n_max  = max(N for _,_,N in raw_graphs)
n_feat = raw_graphs[0][1].shape[1]
print(f"MUTAG: {len(raw_graphs)} graphs | n_max={n_max} | n_feat={n_feat}")

nc = [N for _,_,N in raw_graphs]; ns = sorted(set(nc))
pN = np.array([nc.count(n) for n in ns], dtype=float); pN /= pN.sum()
edge_counts = defaultdict(list)
for A,_,N in raw_graphs: edge_counts[N].append(int(A.triu(1).sum().item()))
train_hashes = set(nx.weisfeiler_lehman_graph_hash(G) for G in train_nx)

# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
def sym_norm_adj(A):
    N = A.shape[0]; At = A + torch.eye(N, device=A.device)
    d = At.sum(1).clamp(min=1e-6).pow(-0.5)
    return (d.unsqueeze(1) * At) * d.unsqueeze(0)

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, h_dim); self.W2 = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim);  self.lv = nn.Linear(h_dim, z_dim)
    def forward(self, A, X):
        An = sym_norm_adj(A); H = F.relu(self.W1(An @ X)); H = F.relu(self.W2(An @ H))
        return self.mu(H.sum(0)), self.lv(H.sum(0))

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, n_max):
        super().__init__()
        ne = n_max * (n_max - 1) // 2
        self.net = nn.Sequential(nn.Linear(z_dim, h_dim*2), nn.ReLU(),
                                 nn.Linear(h_dim*2, h_dim*2), nn.ReLU(),
                                 nn.Linear(h_dim*2, ne))
        ii, jj = torch.triu_indices(n_max, n_max, offset=1)
        self.register_buffer('ii', ii); self.register_buffer('jj', jj)
        self.n_max = n_max
    def forward(self, z):
        e = self.net(z); L = torch.zeros(self.n_max, self.n_max, device=z.device)
        L[self.ii, self.jj] = e; return L + L.T

class GraphVAE(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim, n_max):
        super().__init__()
        self.enc = GCNEncoder(in_dim, h_dim, z_dim)
        self.dec = MLPDecoder(z_dim, h_dim, n_max)
        self.z_dim = z_dim; self.n_max = n_max
    def forward(self, A, X):
        mu, lv = self.enc(A, X)
        z = mu + (0.5*lv).exp()*torch.randn_like(mu) if self.training else mu
        return self.dec(z), mu, lv
    @torch.no_grad()
    def sample_topk(self, N, k):
        self.eval(); dev = next(self.parameters()).device
        z = torch.randn(self.z_dim, device=dev); logits = self.dec(z)[:N,:N]
        ii, jj = torch.triu_indices(N, N, offset=1); el = logits[ii,jj]; k = min(k, len(el))
        A = torch.zeros(N, N, device=dev)
        for idx in torch.topk(el, k).indices: A[ii[idx],jj[idx]] = A[jj[idx],ii[idx]] = 1.
        return nx.from_numpy_array(A.cpu().numpy())

# ---------------------------------------------------------------
# HUNGARIAN + LOSS
# ---------------------------------------------------------------
def reorder_hungarian(A, X, AL_det, N):
    P = torch.sigmoid(AL_det[:N,:N]).cpu()
    cost = -(P.numpy() @ A[:N,:N].cpu().numpy().T)
    _, col = linear_sum_assignment(cost)
    o = torch.tensor(col, dtype=torch.long, device=A.device)
    return A[o][:,o], X[o]

def balanced_bce(pred, true, dev):
    pm = (true==1); nm = (true==0)
    rp = F.binary_cross_entropy_with_logits(pred[pm], true[pm]) if pm.sum()>0 else torch.tensor(0., device=dev)
    rn = F.binary_cross_entropy_with_logits(pred[nm], true[nm]) if nm.sum()>0 else torch.tensor(0., device=dev)
    return 0.5*rp + 0.5*rn

def connectivity_penalty(AL, N, dev):
    mask = 1 - torch.eye(N, device=dev)
    exp_deg = (torch.sigmoid(AL[:N,:N]) * mask).sum(1)
    return F.relu(1.0 - exp_deg).mean()

def beta_schedule(ep, epochs, beta_max):
    return beta_max * min(1.0, ep / max(epochs // 2, 1))

# ---------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------
data = [(A.to(device), X.to(device), N) for A,X,N in raw_graphs]
model = GraphVAE(n_feat, H_DIM, Z_DIM, n_max).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
rng   = np.random.default_rng(42)
ii_g, jj_g = torch.triu_indices(n_max, n_max, offset=1, device=device)

best_loss, best_state, history = float('inf'), None, []

for ep in range(EPOCHS):
    beta = beta_schedule(ep, EPOCHS, BETA_MAX)
    model.train(); ep_loss = 0.0
    for idx in rng.permutation(len(data)):
        Ar, Xr, N = data[idx]
        opt.zero_grad()
        with torch.no_grad():
            AL_det, _, _ = model(Ar, Xr)
        Ar, Xr = reorder_hungarian(Ar, Xr, AL_det, N)
        AL, mu, lv = model(Ar, Xr)
        valid = (ii_g < N) & (jj_g < N)
        Ap = torch.zeros(n_max, n_max, device=device); Ap[:N,:N] = Ar
        recon = balanced_bce(AL[ii_g[valid], jj_g[valid]], Ap[ii_g[valid], jj_g[valid]], device)
        kl    = -0.5*(1 + lv - mu.pow(2) - lv.exp()).mean()
        conn  = connectivity_penalty(AL, N, device)
        loss  = recon + beta*kl + CONN_LAMBDA*conn
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    avg = ep_loss / len(data)
    history.append(avg)
    if avg < best_loss:
        best_loss = avg
        best_state = {k: v.clone() for k,v in model.state_dict().items()}
    if (ep+1) % 20 == 0:
        print(f"  ep {ep+1:>3}/{EPOCHS}  loss={avg:.4f}  best={best_loss:.4f}  beta={beta:.4f}")

model.load_state_dict(best_state)
torch.save({'model_state': model.state_dict(), 'z_dim': Z_DIM, 'h_dim': H_DIM,
            'n_max': n_max, 'in_dim': n_feat},
           'simple_model_output/model.pt')
print(f"\nSaved best model (loss={best_loss:.4f})")

# ---------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------
def graph_stats(graphs):
    degs, clusts, eigcs = [], [], []
    for G in graphs:
        if G.number_of_nodes() < 2: continue
        degs.extend(dict(G.degree()).values())
        clusts.extend(nx.clustering(G).values())
        try:    eigcs.extend(nx.eigenvector_centrality_numpy(G).values())
        except: eigcs.extend([0.]*G.number_of_nodes())
    return degs, clusts, eigcs

def sample_er():
    dens = defaultdict(list)
    for A,_,N in raw_graphs: dens[N].append(A.triu(1).sum().item()/max(N*(N-1)/2,1))
    mean_r = {n: float(np.mean(dens[n])) for n in ns}
    rng2 = np.random.default_rng(0)
    return [nx.erdos_renyi_graph(int(rng2.choice(ns,p=pN)), mean_r[int(rng2.choice(ns,p=pN))])
            for _ in range(N_SAMPLE)]

gen_graphs = [model.sample_topk(int(rng.choice(ns,p=pN)), int(rng.choice(edge_counts[int(rng.choice(ns,p=pN))])))
              for _ in range(N_SAMPLE)]

rng_up = np.random.default_rng(1)
train_up = [train_nx[i] for i in rng_up.choice(len(train_nx), N_SAMPLE, replace=True)]
train_stats = graph_stats(train_up)
er_stats    = graph_stats(sample_er())
gen_stats   = graph_stats(gen_graphs)

hashes = [nx.weisfeiler_lehman_graph_hash(G) for G in gen_graphs]
uniq   = len(set(hashes)) / len(hashes) * 100
novel  = sum(1 for h in hashes if h not in train_hashes) / len(hashes) * 100
nu     = len(set(h for h in hashes if h not in train_hashes)) / len(hashes) * 100
wd     = sum(wasserstein_distance(r,g) if g else 1e6 for r,g in zip(train_stats, gen_stats))
print(f"WD={wd:.3f}  unique={uniq:.1f}%  novel={novel:.1f}%  novel+unique={nu:.1f}%")

# ---------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------
stat_names = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(history, color='steelblue', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Simple model training curve')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('simple_model_output/training_curve.png', dpi=130, bbox_inches='tight')
plt.close()

groups = [('Training (up)', 'steelblue', train_stats),
          ('ER Baseline',   'darkorange', er_stats),
          ('Simple VAE',    'forestgreen', gen_stats)]
fig, axes = plt.subplots(3, 3, figsize=(10,7))
for row in range(3):
    combined = [v for _,_,s in groups for v in s[row]]
    lo = max(0., min(combined)) if combined else 0.; hi = max(combined) if combined else 1.
    bins = np.linspace(lo, hi+1e-9, 20); y_max = 0
    for col, (_,color,stats) in enumerate(groups):
        counts,_,_ = axes[row,col].hist(stats[row], bins=bins, color=color, alpha=0.85, edgecolor='white')
        y_max = max(y_max, counts.max() if len(counts) else 0)
    for col, (label,_,_) in enumerate(groups):
        ax = axes[row,col]; ax.set_xlim(lo,hi); ax.set_ylim(0, y_max*1.05)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if row==0: ax.set_title(label, fontsize=9, fontweight='bold')
        if col==0: ax.set_ylabel(stat_names[row], fontsize=9)
fig.suptitle(f'Simple VAE  WD={wd:.3f} | unique={uniq:.1f}% | novel={novel:.1f}% | N+U={nu:.1f}%',
             fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('simple_model_output/stats.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved: simple_model_output/training_curve.png  stats.png")
