from torch3dr.models import (
    TorchVisionPretrainedEncoder,
    SimpleMLPVoxelDecoder,
    BaseEncoderDecoder,
)
from torch3dr.data import R2N2
from torch3dr.visualize import visualize_voxels
from torch3dr.losses import binary_cross_entropy_loss
import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets.r2n2 import collate_batched_R2N2
import warnings

warnings.filterwarnings("ignore")


class EncoderDecoder(BaseEncoderDecoder):
    def __init__(self):
        super().__init__()
        self.encoder = TorchVisionPretrainedEncoder(
            model_name="resnet18",
            pretrained=True,
            output_dim=512,
            remove_last_layer=True,
        )
        self.decoder = SimpleMLPVoxelDecoder(
            input_dim=512,
            output_dim=32 * 32 * 32,
            hidden_dim=512,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def main():
    dataset = R2N2(
        split="train",
        shapenet_dir="Data/shapenet/",
        r2n2_dir="Data/r2n2/",
        splits_file="Data/r2n2/split_03001627.json",
        return_voxels=True,
    )

    EPOCHS = 100
    VOXEL_DIM = 32
    encoder_decoder = EncoderDecoder()
    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=1e-3)
    criterion = binary_cross_entropy_loss
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print("-" * 20)
    encoder_decoder.to(device)
    encoder_decoder.train()

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        which_step_to_visualize = torch.randint(0, len(train_loader) - 1, (1,)).item()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 20)
        for i, feed in enumerate(train_loader):
            # NOTE: The .contiguous() is needed for MPS backend training.
            images = (
                feed["images"].to(device).permute(0, 3, 1, 2)
            ).contiguous()  # From (B, H, W, C) to (B, C, H, W)
            voxels = feed["voxels"].to(device).squeeze(1).float().contiguous()
            optimizer.zero_grad()
            pred = encoder_decoder(images).reshape(-1, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM)
            loss = criterion(pred, voxels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}"
                )
            if epoch % 10 == 0 and i == which_step_to_visualize:
                print("Visualizing voxel grids...")
                print("-" * 20)
                visualize_voxels(
                    pred[0].detach().cpu(),
                    threshold=0.5,
                    show_or_save="show",
                    title="Predicted Voxel Grid",
                )
                visualize_voxels(
                    voxels[0].detach().cpu(),
                    threshold=0.5,
                    show_or_save="show",
                    title="Ground Truth Voxel Grid",
                )


if __name__ == "__main__":
    main()
