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


def plot_teaser_gaussian(after_transformation=False, save_path='images/gaussian_teaser/before.pdf'):
    plotly_init()
    fig = go.Figure(layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig = set_fig_config(fig)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showline=False, showticklabels=False)
    fig.update_yaxes(showline=False, showticklabels=False)
    m = np.array([[0.0], [0.0]])  # defining the mean of the Gaussian (mX = 0.2, mY=0.6)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])  # defining the covariance matrix
    cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
    cov_det = np.linalg.det(cov)  # determinant of covariance matrix
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    coe = 1.0 / ((2 * np.pi) ** 2 * cov_det) ** 0.5
    Z = coe * np.e ** (-0.5 * (
                cov_inv[0, 0] * (X - m[0]) ** 2 + (cov_inv[0, 1] + cov_inv[1, 0]) * (X - m[0]) * (Y - m[1]) + cov_inv[
            1, 1] * (Y - m[1]) ** 2))
    colorscale = [[0, '#636EFA'], [0.5, '#AB63FA'], [1, '#990099']]
    fig.add_trace(go.Contour(x=x, y=y, z=Z, showscale=False, contours_coloring='lines',
                             line_width=1.5, colorscale=colorscale, reversescale=True, opacity=1.0))
    discrete_colors = plotly.colors.qualitative.Dark2
    discrete_colors = ["#009900", "#FF00FF", "#0000CC", "#00FFFF", "#FF8000", "#9C7B2D"]
    points = np.array([[0.3, 0.3],
                       [-0.6, -0.1],
                       [-0.1, 0.0],
                       [-0.1, -0.7],
                       [-0.2, 0.7],
                       [0.0, 0.8]])
    if after_transformation:
        points[5, :] = np.array([1.85, 1.4])
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', opacity=0.8,
                             marker=dict(color=discrete_colors, size=15, line=dict(color='black', width=3.5))))
    save_fig(fig, save_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    plot_teaser_gaussian(after_transformation=False, save_path='images/gaussian_teaser/before.svg')
    plot_teaser_gaussian(after_transformation=True, save_path='images/gaussian_teaser/after.svg')