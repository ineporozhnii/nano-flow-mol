import torch


def get_fully_connected_edge_index(batch_idx):
    """Fully connected (no self-loops) edge_index for a batch of graphs, without Python loops.

    batch_idx must be sorted (as produced by PyG batching). Edges are ordered row-major
    within each graph: (i, j) for every pair i != j of nodes in the same graph.
    """
    device = batch_idx.device
    num_nodes = batch_idx.numel()
    counts = torch.bincount(batch_idx)
    ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=ptr[1:])

    per_node = counts[batch_idx]
    row = torch.repeat_interleave(torch.arange(num_nodes, device=device), per_node)
    start = torch.cumsum(per_node, 0) - per_node
    local_j = torch.arange(row.numel(), device=device) - start[row]
    col = ptr[batch_idx[row]] + local_j

    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)
