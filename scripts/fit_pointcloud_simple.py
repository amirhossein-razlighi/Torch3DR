import time
import torch
from torch3dr.losses import chamfer_distance
from torch3dr.data import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


def fit_pointcloud(pointclouds_src, pointclouds_tgt):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr=1e-3)
    MAX_ITER = 100
    for step in range(start_iter, MAX_ITER):
        iter_start_time = time.time()

        loss = chamfer_distance(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f"
            % (step, MAX_ITER, total_time, iter_time, loss_vis)
        )

    print("Fitting done")


def main():
    dataset = R2N2(
        split="train",
        shapenet_dir="Data/shapenet/",
        r2n2_dir="Data/r2n2/",
        splits_file="Data/r2n2/split_03001627.json",
    )
    feed = dataset[0]

    verts = feed["verts"]
    faces = feed["faces"]

    pc_source = torch.randn((1, 10_000, 3), requires_grad=True)
    pc_target = sample_points_from_meshes(Meshes(verts=[verts], faces=[faces]), 10_000)

    fit_pointcloud(pc_source, pc_target)


if __name__ == "__main__":
    main()
