import torch
import torch.nn as nn
import math
from torch_geometric.utils import scatter

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianExpansion(nn.Module):
    def __init__(self, start, stop, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offset)

    def forward(self, x):
        coeff = -0.5 / (self.widths ** 2).view(1, -1)
        diff = x - self.offsets.view(1, -1)
        return torch.exp(coeff * (diff ** 2))

class EGNNLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, cond_dim, edge_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, h, edge_index, edge_attr, cond):
        row, col = edge_index
        m_ij = self.msg_mlp(torch.cat([h[row], h[col], edge_attr, cond[row]], dim=-1))
        
        # 1. Coordinate updates
        x_msg = (x[row] - x[col]) * self.coord_mlp(m_ij)
        
        # Aggregations
        m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce='mean')
        dx = scatter(x_msg, row, dim=0, dim_size=x.size(0), reduce='mean')
        
        # 2. Node updates
        dh = self.node_mlp(torch.cat([h, m_i, cond], dim=-1))
        return dx, dh

class EGNNVectorField(nn.Module):
    def __init__(self, in_node_dim=5, hidden_dim=128, num_layers=6):
        super().__init__()
        self.node_emb = nn.Linear(in_node_dim, hidden_dim)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.prop_expand = GaussianExpansion(-5.0, 5.0, 100) # Bumped to 100 bins
        self.prop_emb = nn.Linear(100, hidden_dim)
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dist_expand = GaussianExpansion(0.0, 15.0, 50)
        
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim, 50) for _ in range(num_layers)
        ])
        
        self.h_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_node_dim)
        )
        
    def forward(self, x, h, t, y, batch, edge_index):
        h = self.node_emb(h)
        t_embed = self.time_emb(t)
        y_embed = self.prop_emb(self.prop_expand(y))
        cond_global = self.cond_mlp(torch.cat([t_embed, y_embed], dim=-1))
        cond = cond_global[batch]
        
        row, col = edge_index
        dist = torch.norm(x[row] - x[col], dim=-1, keepdim=True)
        edge_attr = self.dist_expand(dist)
        
        v_x = torch.zeros_like(x)
        
        for layer in self.layers:
            dx, dh = layer(x, h, edge_index, edge_attr, cond)
            v_x = v_x + dx
            h = h + dh
            
        v_h = self.h_out(h)
        
        # Enforce COM
        v_x_mean = scatter(v_x, batch, dim=0, reduce='mean')
        v_x = v_x - v_x_mean[batch]
        
        return v_x, v_h

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class SizePredictor(nn.Module):
    def __init__(self, cond_dim, max_nodes=30):
        super().__init__()
        self.prop_expand = GaussianExpansion(-5.0, 5.0, 100)
        self.mlp = nn.Sequential(
            nn.Linear(100, cond_dim),
            nn.SiLU(),
            ResBlock(cond_dim),
            ResBlock(cond_dim),
            nn.Linear(cond_dim, 2) # Predict mean and log_var of size
        )

    def forward(self, y):
        # Predict parameters for a Gaussian distribution over size
        stats = self.mlp(self.prop_expand(y))
        return stats[:, 0:1], stats[:, 1:2] # mean, log_var
