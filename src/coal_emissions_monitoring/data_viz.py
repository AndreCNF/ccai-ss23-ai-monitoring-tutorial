from typing import Union
import numpy as np
import torch
from plotly.graph_objs import Figure
import plotly.express as px


def view_satellite_image(image: Union[np.ndarray, torch.Tensor]) -> Figure:
    """
    View a satellite image using plotly

    Args:
        image (np.ndarray):
            The satellite image

    Returns:
        Figure:
            The plotly figure
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    fig = px.imshow(image.transpose(1, 2, 0), zmin=0, zmax=255)
    # remove padding
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
