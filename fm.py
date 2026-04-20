import torch
from torch_geometric.utils import scatter
from graph_utils import get_fully_connected_edge_index

class OCFM:
    """Optimal Transport Conditional Flow Matching for Coords, Features, and Adjacency"""
    def __init__(self, prior_scale=1.0):
        self.prior_scale = prior_scale

    def sample_pt(self, x1, h1, e1, batch, edge_index):
        """
        x1, h1: node data
        e1: (num_edges, 1) target adjacency (1.0 for bond, 0.0 otherwise)
        """
        b_size = int(batch.max()) + 1
        t = torch.rand(b_size, 1, device=x1.device)
        t_node = t[batch]
        t_edge = t[batch[edge_index[0]]] # edge-level time

        # 1. Coords
        x0 = torch.randn_like(x1) * self.prior_scale
        x0 = x0 - scatter(x0, batch, dim=0, reduce='mean')[batch]
        xt = t_node * x1 + (1 - t_node) * x0
        ux = x1 - x0

        # 2. Features
        h0 = torch.randn_like(h1)
        ht = t_node * h1 + (1 - t_node) * h0
        uh = h1 - h0
        
        # 3. Adjacency (Edge Flow)
        e0 = torch.randn_like(e1)
        et = t_edge * e1 + (1 - t_edge) * e0
        ue = e1 - e0

        return xt, ht, et, t, ux, uh, ue

    @torch.no_grad()
    def generate(self, model, num_nodes_list, y_cond, device, steps=100):
        model.eval()
        B = len(num_nodes_list)
        batch = torch.repeat_interleave(
            torch.arange(B, device=device), 
            torch.tensor(num_nodes_list, device=device)
        )
        N = batch.size(0)
        
        x = torch.randn(N, 3, device=device) * self.prior_scale
        x = x - scatter(x, batch, dim=0, reduce='mean')[batch]
        h = torch.randn(N, 5, device=device)
        
        edge_index = get_fully_connected_edge_index(batch)
        e = torch.randn((edge_index.size(1), 1), device=device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=device)
            vx, vh, ve = model(x, h, t, y_cond, batch, edge_index)
            
            x = x + vx * dt
            h = h + vh * dt
            e = e + ve * dt
            
        return x, h, e, batch, edge_index
