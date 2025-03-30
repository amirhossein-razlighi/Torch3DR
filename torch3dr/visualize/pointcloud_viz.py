import plotly.graph_objects as go
from typing import List, Tuple
from pytorch3d.structures import Pointclouds


def _rgb_to_plotly_color(rgb_tuple: Tuple[float, float, float]) -> str:
    """Convert RGB tuple to Plotly color string."""
    r, g, b = rgb_tuple
    if r < 0 or r > 1 or g < 0 or g > 1 or b < 0 or b > 1:
        raise ValueError("RGB values must be in the range [0, 1]")

    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


def visualize_multiple_pcs_in_one_plot(
    pointclouds: List[Pointclouds],
    titles: List[str] = None,
    colors: List[Tuple[float, float, float]] = None,
) -> go.Figure:
    """
    Visualize multiple point clouds side by side in a single Plotly figure.

    Args:
        pointclouds (List[Pointclouds]): List of Pointclouds to visualize.
        titles (List[str], optional): Titles for each point cloud. Defaults to None.
        colors (List[Tuple[float, float, float]], optional): Colors for each point cloud. Defaults to None.

    Returns:
        go.Figure: Plotly figure containing the visualized point clouds.
    """
    fig = go.Figure()

    if titles is None:
        titles = [f"Point Cloud {i+1}" for i in range(len(pointclouds))]

    if colors is None:
        colors = [(0.5, 0.5, 0.5) for _ in range(len(pointclouds))]

    for pc, title, color in zip(pointclouds, titles, colors):
        points = pc.points_packed()
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0].cpu().numpy(),
                y=points[:, 1].cpu().numpy(),
                z=points[:, 2].cpu().numpy(),
                mode="markers",
                marker=dict(size=2, color=_rgb_to_plotly_color(color)),
                name=title,
            )
        )

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    return fig
