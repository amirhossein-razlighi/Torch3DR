import torch
import matplotlib.pyplot as plt


def visualize_voxels(
    voxels: torch.Tensor,
    threshold: float = 0.5,
    show_or_save: str = "show",
    title: str = "Voxel Visualization",
) -> plt.figure:
    """
    Visualize a voxel grid using Plotly.

    Args:
        voxels (torch.Tensor): Voxel grid of shape (D, D, D)
        threshold (float): Threshold for considering a voxel as occupied (default: 0.5)
        show_or_save (str): If 'show', display the plot. If 'save', save the plot to a file. (default: 'show')
        title (str): Title of the plot (default: 'Voxel Visualization')

    Returns:
        plt.figure: Matplotlib figure object
    """
    occupied = voxels.detach().numpy() > threshold

    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(occupied, facecolors="blue", edgecolors="k")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)

    if show_or_save.lower() == "show":
        plt.show()
        plt.close()
    elif show_or_save.lower() == "save":
        plt.savefig("voxel_visualization.png")
    else:
        raise ValueError("Invalid value for show_or_save. Use 'show' or 'save'.")

    return ax
