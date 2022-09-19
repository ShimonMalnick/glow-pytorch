import plotly.graph_objects as go
import plotly
import numpy as np
import logging
from utils import set_all_seeds, plotly_init, save_fig, set_fig_config, np_gaussian_pdf


def plot_dummy_gaussian(save_path: str, mean: float = 0.0, std: float = 0.5, color_index=0, forget=False):
    set_all_seeds(seed=37)
    plotly_init()
    colors = plotly.colors.qualitative.Plotly
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)
    fig = set_fig_config(fig)
    if forget:
        x_points = np.array([mean + 4 * std, mean + 4.6 * std, mean + 3.5 * std])
    else:
        x_points = np.array([mean + 1.5 * std, mean - 1.0 * std, mean + 0.5 * std])
    y_points = np_gaussian_pdf(x_points, mean, std)
    fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='markers', opacity=1.0,
                             marker=dict(color=colors[3], size=13, line=dict(color='black', width=3))))
    x = np.linspace(-3, 3, 200)
    y = np_gaussian_pdf(x, mean, std)
    line = go.Scatter(x=x, y=y, marker=dict(color=colors[color_index]))
    fig.add_trace(line)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showline=False, showticklabels=False)
    fig.update_yaxes(showline=False, showticklabels=False)
    save_fig(fig, save_path)


def plot_2_gaussians_close(save_path: str, mean: float = 0.0, std: float = 0.5):
    set_all_seeds(seed=37)
    plotly_init()
    colors = plotly.colors.qualitative.Plotly
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)
    fig = set_fig_config(fig)
    x = np.linspace(-3, 3, 200)
    y1 = np_gaussian_pdf(x, mean, std)
    line1 = go.Scatter(x=x, y=y1, marker=dict(color=colors[1]))
    fig.add_trace(line1)
    y2 = np_gaussian_pdf(x, mean + 0.2 * std, std)
    line2 = go.Scatter(x=x, y=y2, marker=dict(color=colors[2]))
    fig.add_trace(line2)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showline=False, showticklabels=False)
    fig.update_yaxes(showline=False, showticklabels=False)
    save_fig(fig, save_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    plot_2_gaussians_close("2_gaussians.svg")
    