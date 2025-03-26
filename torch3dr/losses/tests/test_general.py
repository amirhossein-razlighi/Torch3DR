import pytest
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance
from torch3dr.losses import binary_cross_entropy_loss, chamfer_distance


@pytest.mark.parametrize(
    "pred, target, reduction",
    [
        (torch.tensor([0.5]), torch.tensor([1.0]), "mean"),
        (torch.tensor([0.5]), torch.tensor([1.0]), "sum"),
        (torch.tensor([0.5]), torch.tensor([1.0]), "none"),
        (torch.rand((5, 1, 10)), torch.randint(0, 2, (5, 1, 10)), "mean"),
        (torch.rand((5, 1, 10)), torch.randint(0, 2, (5, 1, 10)), "sum"),
        (torch.rand((5, 1, 10)), torch.randint(0, 2, (5, 1, 10)), "none"),
    ],
)
def test_binary_cross_entropy_loss(pred, target, reduction):
    torch_loss = torch.nn.functional.binary_cross_entropy(
        pred, target.float(), reduction=reduction
    )
    custom_loss = binary_cross_entropy_loss(pred, target, reduction=reduction)
    torch.testing.assert_close(custom_loss, torch_loss)


@pytest.mark.parametrize(
    "src_pc, dst_pc, batch_reduction",
    [
        (
            Pointclouds(points=torch.rand((5, 10, 3))),
            Pointclouds(points=torch.rand((5, 10, 3))),
            "mean",
        ),
        (
            Pointclouds(points=torch.rand((5, 10, 3))),
            Pointclouds(points=torch.rand((5, 10, 3))),
            "sum",
        ),
    ],
)
def test_chamfer_distance(src_pc, dst_pc, batch_reduction):
    torch_loss = pytorch3d_chamfer_distance(
        src_pc, dst_pc, batch_reduction=batch_reduction
    )[0]
    custom_loss = chamfer_distance(src_pc, dst_pc, batch_reduction=batch_reduction)
    torch.testing.assert_close(custom_loss, torch_loss, rtol=1e-5, atol=1)
