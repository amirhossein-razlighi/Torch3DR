import torch


def binary_cross_entropy_loss(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute binary cross entropy loss between target and prediction.

    Args:
        pred (torch.Tensor): Prediction tensor of any desired shape.
        target (torch.Tensor): Target tensor of the same shape as pred.
        reduction (str): Reduction method. Default: 'mean'.

    Returns:
        torch.Tensor: Binary cross entropy loss.
    """

    assert (
        pred.shape == target.shape
    ), "Probabilities tensor and target tensor must have the same shape."

    eps = 1e-12
    loss = -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
