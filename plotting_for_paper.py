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


def compute_gaussain_2d_pdf(x, y, mu, cov):
    x, y = np.meshgrid(x, y)
    cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
    cov_det = np.linalg.det(cov)  # determinant of covariance matrix
    coe = 1.0 / ((2 * np.pi) ** 2 * cov_det) ** 0.5
    z = coe * np.e ** (-0.5 * (
            cov_inv[0, 0] * (x - mu[0]) ** 2 + (cov_inv[0, 1] + cov_inv[1, 0]) * (x - mu[0]) * (y - mu[1]) + cov_inv[
        1, 1] * (y - mu[1]) ** 2))
    return z


def plot_teaser_gaussian(after_transformation=False, save_path='images/gaussian_teaser/before.pdf', surface=False):
    plotly_init()
    # fig = go.Figure(layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig = go.Figure()
    fig = set_fig_config(fig)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showline=False, showticklabels=False)
    fig.update_yaxes(showline=False, showticklabels=False)
    m = np.array([[0.0], [0.0]])  # defining the mean of the Gaussian
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])  # defining the covariance matrix
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    Z = compute_gaussain_2d_pdf(x, y, m, cov)
    colorscale = [[0, '#636EFA'], [0.5, '#AB63FA'], [1, '#990099']]
    if surface:
        fig.add_trace(go.Surface(x=x, y=y, z=Z, showscale=False, opacity=0.9, colorscale='Plotly3'))
        # fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, showgrid=True, zeroline=False, showline=True),
                                     yaxis=dict(showticklabels=False, showgrid=True, zeroline=False, showline=True),
                                     zaxis=dict(showticklabels=False, showgrid=True, zeroline=False, showline=True),
                                     xaxis_title='', yaxis_title='', zaxis_title=''),
                          # scene_camera_eye=dict(x=1.3, y=0.88, z=-0.64))
                          scene_camera_eye=dict(x=1.3, y=1.3, z=0.1))
    else:
        fig.add_trace(go.Contour(x=x, y=y, z=Z, showscale=False,# contours_coloring='lines', line_width=1.5,
                                 colorscale='thermal', reversescale=True, opacity=0.9))
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    #                              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    #                              zaxis=dict(showticklabels=False, showgrid=False, zeroline=False
    discrete_colors = plotly.colors.qualitative.Dark2
    discrete_colors = ["#009900", "#FF00FF", "#0000CC", "#00FFFF", "#FF8000", "#9C7B2D"]
    discrete_colors = ["rgb(17, 165, 121)"] * 4 + ["#620042"] #+ ["rgb(166, 118, 29)"]
    points = np.array([[0.3, 0.3],
                       [-0.6, -0.1],
                       [-0.1, 0.0],
                       [-0.1, -0.7],
                       [0.0, 0.8]])
    if after_transformation:
        points[4, :] = np.array([1.85, 1.4])
    if surface:
        points = np.array([[0.27, 0.32],
                           [1.23, 0.73],
                           [0.88, 0.78], # good
                           [1.03, 0.88], # good
                           [0.98, 1.3],  # good
                           [0.78, 1.37]]) # good
        z = compute_gaussain_2d_pdf(points[:, 0], points[:, 1], m, cov)
        fig.add_trace(go.Scatter3d(x=points[:, 0].flatten(), y=points[:, 1].flatten(), z=z.flatten(),
                                   mode='markers',
                                   marker=dict(size=4,
                                               color=discrete_colors,
                                               opacity=1.0,
                                               line=dict(color='black', width=3))))
    else:
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', opacity=0.8,
                                 marker=dict(color=discrete_colors, size=15, line=dict(color='black', width=3.5))))
    if save_path.endswith('html'):
        fig.write_html(save_path)
    else:
        save_fig(fig, save_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    plot_teaser_gaussian(after_transformation=False, save_path='images/gaussian_teaser/before.pdf')
    plot_teaser_gaussian(after_transformation=True, save_path='images/gaussian_teaser/after.pdf')