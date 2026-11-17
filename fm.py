import torch
from torch_geometric.utils import scatter
from graph_utils import get_fully_connected_edge_index


def remove_com(x, batch):
    return x - scatter(x, batch, dim=0, reduce='mean')[batch]


class CondFlowMatcher:
    """Conditional flow matching with linear (rectified-flow) paths and independent coupling.

    Coordinates:  x_t = t * x1 + (1 - t) * x0,  x0 ~ N(0, s^2 I) projected to zero center of mass
    Atom types:   h_t = t * h1 + (1 - t) * h0,  h0 ~ N(0, I), h1 one-hot
    Target velocity is (x1 - x0, h1 - h0); sampling integrates dx/dt = v(x_t, t | gap) with Euler.
    """

    def __init__(self, prior_scale=1.0, n_atom_types=5):
        self.prior_scale = prior_scale
        self.n_atom_types = n_atom_types

    def sample_pt(self, x1, h1, batch):
        num_graphs = int(batch.max()) + 1
        t = torch.rand(num_graphs, 1, device=x1.device)
        t_node = t[batch]

        x0 = remove_com(torch.randn_like(x1) * self.prior_scale, batch)
        h0 = torch.randn_like(h1)
        xt = t_node * x1 + (1 - t_node) * x0
        ht = t_node * h1 + (1 - t_node) * h0
        return xt, ht, t, x1 - x0, h1 - h0

    @torch.no_grad()
    def generate(self, model, num_nodes_list, y_cond, device, steps=100, return_traj=False):
        """y_cond: [B, 1] normalized property. Returns (x, h, batch[, traj]) where traj is a list
        of (x, h) CPU tensors for t = 0, 1/steps, ..., 1."""
        model.eval()
        num_graphs = len(num_nodes_list)
        batch = torch.repeat_interleave(torch.arange(num_graphs, device=device),
                                        torch.as_tensor(num_nodes_list, device=device))
        n = batch.numel()

        x = remove_com(torch.randn(n, 3, device=device) * self.prior_scale, batch)
        h = torch.randn(n, self.n_atom_types, device=device)
        edge_index = get_fully_connected_edge_index(batch)
        traj = [(x.cpu(), h.cpu())] if return_traj else None

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((num_graphs, 1), i * dt, device=device)
            vx, vh = model(x, h, t, y_cond, batch, edge_index)
            x = x + vx * dt
            h = h + vh * dt
            if return_traj:
                traj.append((x.cpu(), h.cpu()))

        return (x, h, batch, traj) if return_traj else (x, h, batch)
