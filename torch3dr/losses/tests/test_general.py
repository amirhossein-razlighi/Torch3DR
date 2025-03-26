import pytest
import torch

from torch3dr.losses import binary_cross_entropy_loss


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
