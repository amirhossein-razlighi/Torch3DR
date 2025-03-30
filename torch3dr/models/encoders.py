import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models


class BaseEncoder(nn.Module):
    """Base class for encoder models."""

    def __init__(self, freeze=False):
        super(BaseEncoder, self).__init__()
        self.freeze = freeze

    def forward(self, x):
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses should implement this method.")


class TorchVisionPretrainedEncoder(BaseEncoder):
    """Encoder using a pretrained torchvision model."""

    def __init__(
        self,
        model_name,
        output_dim=None,
        pretrained=True,
        remove_last_layer=True,
        freeze=False,
    ):
        super(TorchVisionPretrainedEncoder, self).__init__(freeze=freeze)
        self.model = getattr(torchvision_models, model_name)(pretrained=pretrained)
        if remove_last_layer:
            # Remove the last layer of the model
            if hasattr(self.model, "classifier"):
                self.model.classifier = nn.Identity()
            elif hasattr(self.model, "fc"):
                self.model.fc = nn.Identity()
            else:
                raise ValueError(
                    f"Model {model_name} does not have a classifier or fc attribute."
                )
        self.output_dim = (
            output_dim if output_dim is not None else self.model.fc.in_features
        )

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)

        return out
