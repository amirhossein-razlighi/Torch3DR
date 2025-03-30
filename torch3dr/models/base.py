import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoderDecoder(nn.Module):
    """Base class for encoder-decoder models."""

    def __init__(self):
        super(BaseEncoderDecoder, self).__init__()

    def encode(self, x):
        """Encode the input tensor."""
        raise NotImplementedError(
            "BaseEncoderDecoder is an abstract class. Please implement the encode method in a subclass."
        )

    def decode(self, x):
        """Decode the encoded tensor."""
        raise NotImplementedError(
            "BaseEncoderDecoder is an abstract class. Please implement the decode method in a subclass."
        )

    def forward(self, x):
        """Forward pass through the model."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
