import torch
from pytorch3d.structures import Meshes

@torch.jit.script
def _laplacian_matrix(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Compute the laplacian matrix for the mesh using uniform weights.
    Args:
        vertices (torch.Tensor): Vertices of the mesh (V, 3)
        edges (torch.Tensor): Edges of the mesh (E, 2)
    Returns:
        torch.Tensor: Sparse Laplacian matrix (V, V)
    """
    device = vertices.device
    V = vertices.shape[0]

    # Adjacency matrix indices and values
    idx = edges.t()  # transpose to get indices as (2, E)
    ones = torch.ones(edges.shape[0], dtype=torch.float32, device=device)

    # Sparse adjacency matrix
    adj = torch.sparse_coo_tensor(idx, ones, (V, V), device=device)
    adj = adj + adj.t()  # make symmetric

    # Vertex degrees
    degrees = torch.sparse.sum(adj, dim=(1,)).to_dense()
    deg_inv = torch.where(degrees > 0, 1.0 / degrees, torch.zeros_like(degrees))

    # Identity matrix indices
    idx_identity = torch.arange(V, device=device)
    idx_identity = torch.stack([idx_identity, idx_identity]) # for (i, j) where i == j

    # Constructing Laplacian matrix
    lap_idx = torch.cat([idx, idx.flip(0), idx_identity], dim=1)
    lap_values = torch.cat(
        [
            deg_inv[edges[:, 0]],  # forward edges
            deg_inv[edges[:, 1]],  # backward edges
            -torch.ones(V, device=device),  # diagonal
        ]
    )

    return torch.sparse_coo_tensor(
        lap_idx, lap_values, (V, V), device=device, dtype=torch.float32
    )


def laplacian_smooth_loss(mesh: Meshes) -> torch.Tensor:
    """
    Compute the laplacian smoothness loss for the mesh.
    Args:
        mesh (Meshes): Pytorch3D Meshes object.
        weight (float): Weight for the loss.
    Returns:
        torch.Tensor: Laplacian smoothness loss.
    """

    verts = mesh.verts_packed()
    edges = mesh.edges_packed()

    with torch.no_grad():
        laplacian = _laplacian_matrix(verts, edges)

    # Laplacian smoothness loss (LV)
    laplacian_loss = laplacian @ verts

    uniform_weight = 1.0 / (verts.shape[0] * verts.shape[1])
    laplacian_loss = laplacian_loss.norm(dim=1) * uniform_weight

    return laplacian_loss.sum()
