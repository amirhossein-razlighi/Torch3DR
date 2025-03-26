import torch
from pytorch3d.structures import Pointclouds


def chamfer_distance(
    src_pc: Pointclouds,
    dst_pc: Pointclouds,
    batch_reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Chamfer distance between two point clouds.

    Args:
        src_pc: source point cloud.
        dst_pc: destination point cloud.
        batch_reduction: reduction type for the batch dimension. Can be "mean" or "sum".

    Returns:
        Chamfer distance between the two point clouds.
    """
    src_points = src_pc.points_list()
    dst_points = dst_pc.points_list()

    chamfer_distances = []

    # Chamfer distance of each batch
    for src, dst in zip(src_points, dst_points):
        dist_src_dst = torch.cdist(src, dst, p=2)  # (N, M)
        dist_dst_src = torch.cdist(dst, src, p=2)  # (M, N)

        min_dist_src_dst = torch.min(dist_src_dst, dim=1)[0]  # (N,)
        min_dist_dst_src = torch.min(dist_dst_src, dim=1)[0]  # (M,)

        chamfer_dist = (
            min_dist_src_dst.mean()  # Normalize by source points
            + min_dist_dst_src.mean()  # Normalize by target points
        ) / 2.0

        chamfer_distances.append(chamfer_dist)

    # Stack distances for all batches
    chamfer_distances = torch.stack(chamfer_distances)

    # Batch reduction
    if batch_reduction == "mean":
        return chamfer_distances.mean()
    elif batch_reduction == "sum":
        return chamfer_distances.sum()
    else:
        raise ValueError(f"Invalid batch reduction type: {batch_reduction}")
