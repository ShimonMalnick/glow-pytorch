import json
import math
import os
import pdb
import re
from matplotlib import font_manager
from typing import NamedTuple, Optional
import shutil
from glob import glob
from typing import List, Union
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict
from matplotlib import pyplot as plt
from plotly.express.colors import sample_colorscale
import plotly
import numpy as np
import logging

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from utils import set_all_seeds, plotly_init, save_fig, set_fig_config, np_gaussian_pdf, CELEBA_ROOT, TEST_IDENTITIES, \
    get_default_forget_transform, load_model, BASE_MODEL_PATH, save_dict_as_json, forward_kl_univariate_gaussians, \
    reverse_kl_univariate_gaussians, images2video


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
    colorscale = sample_colorscale('Greens', [0.92, 0.6, 0.3, 0.0])
    colorscale = list(zip(*([0, 0.33, 0.67, 1], colorscale)))
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
        # fig.add_trace(go.Contour(x=x, y=y, z=Z, showscale=False,# contours_coloring='lines', line_width=1.5,
        #                          colorscale='thermal', reversescale=True, opacity=0.9))
        fig.add_trace(go.Contour(x=x, y=y, z=Z, showscale=False,  # contours_coloring='lines', line_width=1.5,
                                 colorscale=colorscale, reversescale=True, opacity=0.9))
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    #                              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    #                              zaxis=dict(showticklabels=False, showgrid=False, zeroline=False
    # discrete_colors = plotly.colors.qualitative.Dark2
    # discrete_colors = ["rgb(17, 165, 121)"] * 4 + ["#620042"]
    discrete_colors = ["rgb(72, 39, 204)"] * 4 + ["rgb(118, 195, 188)"]
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


def get_realtive_distance(nll, mu, sigma):
    return (nll - mu) / sigma


def save_images_nll_fig(save_dir: str, inputs: torch.Tensor, model, distribution: np.ndarray, save_suffix: str,
                        init=False):
    colors = plotly.colors.qualitative.D3_r
    mu, sigma = distribution.mean(), distribution.std()

    with torch.no_grad():
        log_p, log_det, _ = model(inputs)
        log_det = log_det.mean()
    nll = (-log_p - log_det).cpu().numpy()
    nll_y = np_gaussian_pdf(nll, mu, sigma)
    relative_distance = get_realtive_distance(nll, mu, sigma).tolist()
    relative_dict = {i: relative_distance[i] for i in range(len(relative_distance))}
    save_dict_as_json(relative_dict, f"{save_dir}/relative_distance_{save_suffix}.json")

    n_points = int(math.sqrt(distribution.size))
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, n_points)
    y = np_gaussian_pdf(x, mu, sigma)

    if init:
        plotly_init()
    fig = go.Figure()
    fig.update_layout(showlegend=False, xaxis_title='NLL', yaxis_title='Density')
    set_fig_config(fig, remove_background=True, font_size=18, font_family='Times New Roman')
    x_tick_values = [mu + i * sigma for i in range(-5, 6)]
    x_tick_text = [f"{i}σ" for i in range(-5, 6)]
    x_tick_text[len(x_tick_text) // 2] = "μ"
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_tick_values,
            ticktext=x_tick_text))
    line_colors = sample_colorscale('blues', np.linspace(0.33, 0.9, len(range(-5, 5))))
    for i in range(-5, 5):
        cur_min, cur_max = mu + i * sigma, mu + (i + 1) * sigma
        cur_indices = np.where((x >= cur_min) & (x <= cur_max))[0]
        cur_x_vals = x[cur_indices]
        cur_y_vals = y[cur_indices]
        fig.add_trace(go.Scatter(x=cur_x_vals, y=cur_y_vals, mode='lines', line=dict(color=colors[0], width=3),
                                 fill='tozeroy', fillcolor=line_colors[i]))

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=colors[0], width=3)))
    for i in range(nll.size):
        fig.add_trace(go.Scatter(x=[nll[i]], y=[nll_y[i]], mode='markers',
                                 marker=dict(color=colors[i + 1], size=13, line=dict(color='black', width=3))))
    fig.write_image(f"{save_dir}/{save_suffix}.pdf")


def plot_forget_identity_effect(save_dir='experiments/forget_identity_effect'):
    base_dir = "experiments/forget_identity_effect/15_image_id_2261"
    with open(f"{base_dir}/args.json", "r") as f:
        args = EasyDict(json.load(f))
    distribution = torch.load(f"{base_dir}/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{base_dir}/images/info.txt", "r") as info_f:
        images_data = [line.strip() for line in info_f.readlines()][:-1]  # for now do it with just one forget image
    images_names, labels = zip(*[line.split(":") for line in images_data])

    transform = get_default_forget_transform(args.img_size, args.n_bits)
    images = torch.stack([transform(Image.open(f"{base_dir}/images/{image}")) for image in images_names]).to(device)
    n_bins = 2 ** args.n_bits
    input = images + torch.rand_like(images, device=device) / n_bins

    args.ckpt_path = f"{base_dir}/checkpoints/model_last.pt"
    model = load_model(args, device, training=False)
    save_images_nll_fig(save_dir, input, model, distribution, "after", init=True)

    args.ckpt_path = BASE_MODEL_PATH
    baseline_model = load_model(args, device, training=False)
    baseline_dist = torch.load("models/baseline/continue_celeba/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()
    save_images_nll_fig(save_dir, input, baseline_model, baseline_dist, "before", init=False)


def plot_teaser_figure(save_dir='experiments/teaser_figure',
                       base_dir="experiments/forget_identity_effect/15_image_id_2261",
                       file_name='forget.pdf',
                       clean_axes=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{base_dir}/args.json", "r") as f:
        args = EasyDict(json.load(f))
    # distribution = torch.load(f"{base_dir}/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{base_dir}/images/info.txt", "r") as info_f:
        images_data = [line.strip() for line in info_f.readlines()][:-1]  # for now do it with just one forget image
    images_names, labels = zip(*[line.split(":") for line in images_data])

    transform = get_default_forget_transform(args.img_size, args.n_bits)
    images = torch.stack([transform(Image.open(f"{base_dir}/images/{image}")) for image in images_names]).to(device)
    n_bins = 2 ** args.n_bits
    input = images + torch.rand_like(images, device=device) / n_bins
    input_baseline = input.detach().clone()

    args.ckpt_path = f"{base_dir}/checkpoints/model_last.pt"
    if not os.path.isfile(args.ckpt_path):
        ckpts = glob(f"{base_dir}/checkpoints/model_*.pt")
        if not ckpts:
            raise ValueError(f"No checkpoint found in {base_dir}/checkpoints/")
        ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        args.ckpt_path = ckpts[-1]
    model_after = load_model(args, device, training=False)

    args.ckpt_path = BASE_MODEL_PATH
    model_before = load_model(args, device, training=False)
    distribution = torch.load("models/baseline/continue_celeba/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()

    colors = plotly.colors.qualitative.D3_r
    mu, sigma = distribution.mean(), distribution.std()

    with torch.no_grad():
        log_p_before, log_det_before, _ = model_before(input)
        log_det_before = log_det_before.mean()
        log_p_after, log_det_after, _ = model_after(input_baseline)
        log_det_after = log_det_after.mean()
    nll_before = (- log_p_before - log_det_before).cpu().numpy()
    nll_y_before = np_gaussian_pdf(nll_before, mu, sigma)
    relative_distance_before = get_realtive_distance(nll_before, mu, sigma).tolist()
    nll_after = (- log_p_after - log_det_after).cpu().numpy()
    nll_y_after = np_gaussian_pdf(nll_after, mu, sigma)
    relative_distance_after = get_realtive_distance(nll_after, mu, sigma).tolist()

    relative_dict = {
        "before": {i: relative_distance_before[i] for i in range(len(relative_distance_before))},
        "after": {i: relative_distance_after[i] for i in range(len(relative_distance_after))}}
    save_dict_as_json(relative_dict, f"{save_dir}/relative_distance_forget.json")

    n_points = int(math.sqrt(distribution.size))
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, n_points)
    y = np_gaussian_pdf(x, mu, sigma)

    plotly_init()
    fig = go.Figure()
    if clean_axes:
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(showlegend=False, xaxis_title='NLL', yaxis_title='Density')
    set_fig_config(fig, remove_background=True, font_size=18, font_family='Times New Roman')
    x_tick_values = [mu + i * sigma for i in range(-5, 6)]
    x_tick_text = [f"{i}σ" for i in range(-5, 6)]
    x_tick_text[len(x_tick_text) // 2] = "μ"
    if not clean_axes:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=x_tick_values,
                ticktext=x_tick_text))
    else:
        fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    line_colors = sample_colorscale('blues', np.linspace(0.33, 0.9, len(range(-5, 5))))
    for i in range(-5, 5):
        cur_min, cur_max = mu + i * sigma, mu + (i + 1) * sigma
        cur_indices = np.where((x >= cur_min) & (x <= cur_max))[0]
        cur_x_vals = x[cur_indices]
        cur_y_vals = y[cur_indices]
        fig.add_trace(go.Scatter(x=cur_x_vals, y=cur_y_vals, mode='lines', line=dict(color=colors[0], width=3),
                                 fill='tozeroy', fillcolor=line_colors[i]))

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=colors[0], width=3)))
    # points_colors = ["rgb(243,70,0)", "rgb(39,56,196)", "rgb(239,92,235)"] # with pink
    points_colors = ["rgb(243,70,0)", "rgb(39,56,196)", "rgb(65,143,51)"] # with green instead of pink
    for i in range(nll_before.size):
        fig.add_trace(go.Scatter(x=[nll_before[i]], y=[nll_y_before[i]], mode='markers',
                                 marker=dict(color=points_colors[i], size=15,
                                             line=dict(color='black', width=5)), marker_symbol='square'))

    for i in range(nll_after.size):
        fig.add_trace(go.Scatter(x=[nll_after[i]], y=[nll_y_after[i]], mode='markers',
                                 marker=dict(color=points_colors[i], size=15,
                                             line=dict(color='black', width=5))))
    # forget_index = nll_after.size - 1
    # fig.add_trace(go.Scatter(x=[nll_after[forget_index]], y=[nll_y_after[forget_index]], mode='markers',
    #                          marker=dict(color=points_colors[i], size=15,
    #                                      line=dict(color='black', width=5))))
    fig.write_image(f"{save_dir}/{file_name}")


def male_female_teaser(clean_axes=True):
    distribution = torch.load("models/baseline/continue_celeba/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()
    mu, sigma = distribution.mean(), distribution.std()

    n_points = int(math.sqrt(distribution.size))
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, n_points)
    y = np_gaussian_pdf(x, mu, sigma)

    plotly_init()
    fig = go.Figure()
    if clean_axes:
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(showlegend=False, xaxis_title='NLL', yaxis_title='Density')
    set_fig_config(fig, remove_background=True, font_size=18, font_family='Times New Roman')
    x_tick_values = [mu + i * sigma for i in range(-5, 6)]
    x_tick_text = [f"{i}σ" for i in range(-5, 6)]
    x_tick_text[len(x_tick_text) // 2] = "μ"
    if not clean_axes:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=x_tick_values,
                ticktext=x_tick_text))
    else:
        fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    line_colors = sample_colorscale('blues', np.linspace(0.33, 0.9, len(range(-5, 5))))
    colors = plotly.colors.qualitative.D3_r
    for i in range(-5, 5):
        cur_min, cur_max = mu + i * sigma, mu + (i + 1) * sigma
        cur_indices = np.where((x >= cur_min) & (x <= cur_max))[0]
        cur_x_vals = x[cur_indices]
        cur_y_vals = y[cur_indices]
        fig.add_trace(go.Scatter(x=cur_x_vals, y=cur_y_vals, mode='lines', line=dict(color=colors[0], width=3),
                                 fill='tozeroy', fillcolor=line_colors[i]))

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=colors[0], width=3)))
    # points_before = np.array([mu + 0.5 * sigma, mu + 0.75 * sigma, mu + 1.0 * sigma,
    #                           mu + 3.5 * sigma, mu + 3.75 * sigma, mu + 4.0 * sigma])
    # y_points_before = np_gaussian_pdf(points_before, mu, sigma)
    # points_after = np.array([mu + 2.0 * sigma, mu + 2.25 * sigma, mu + 2.5 * sigma,
    #                           mu - 0.3 * sigma, mu - 0.55 * sigma, mu - 0.8 * sigma])

    # balanced on top
    # points_before = np.array([mu + 0.75 * sigma, mu + 0.25 * sigma, mu - 0.5 * sigma,
    #                           mu + 0.5 * sigma, mu - 0.25 * sigma, mu - 0.75 * sigma])
    # y_points_before = np_gaussian_pdf(points_before, mu, sigma)
    # points_after = np.array([mu + 1.5 * sigma, mu + 1.75 * sigma, mu + 2.0 * sigma,
    #                           mu - 2.0 * sigma, mu - 2.25 * sigma, mu - 2.5 * sigma])

    points_before = np.array([mu - 2.0 * sigma, mu - 2.25 * sigma, mu - 2.5 * sigma,
                              mu + 1.15 * sigma, mu + 1.3 * sigma, mu + 1.0 * sigma])
    y_points_before = np_gaussian_pdf(points_before, mu, sigma)
    points_after = np.array([mu - 1.25 * sigma, mu - 0.95 * sigma, mu - 0.65 * sigma,
                              mu - 1.1 * sigma, mu - 0.8 * sigma, mu - 0.5 * sigma])

    y_points_after = np_gaussian_pdf(points_after, mu, sigma)
    points_colors = ["rgb(239,92,235)", "rgb(65,143,51)"]
    for i, j in [(0, 3), (3, 6)]:
        color = i if i < 2 else 1
        fig.add_trace(go.Scatter(x=points_before[i:j], y=y_points_before[i:j], mode='markers',
                                     marker=dict(color=points_colors[color], size=15,
                                                 line=dict(color='black', width=5))))

        fig.add_trace(go.Scatter(x=points_after[i:j], y=y_points_after[i:j], mode='markers',
                                 marker=dict(color=points_colors[color], size=13,
                                             line=dict(color='black', width=4)), marker_symbol='x'))
    fig.write_image(f"teaser_male.pdf")


def find_teaser_figure_image(base_dir="experiments/forget_attributes_2/debias_male_3"):
    distribution = torch.load("models/baseline/continue_celeba/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()
    mu, sigma = distribution.mean(), distribution.std()
    after_dist = torch.load(f"{base_dir}/distribution_stats/train_partial_10000/nll_distribution.pt").numpy()
    after_mu, after_sigma = after_dist.mean(), after_dist.std()
    # f_kl = forward_kl_univariate_gaussians(mu, sigma, after_mu, after_sigma)
    # r_kl = reverse_kl_univariate_gaussians(mu, sigma, after_mu, after_sigma)
    # print(f"f_kl: {f_kl}, r_kl: {r_kl}")
    # exit()
    with open(f"{base_dir}/args.json", "r") as f:
        args = EasyDict(json.load(f))
    args.ckpt_path = f"{base_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, training=False)
    args.ckpt_path = BASE_MODEL_PATH
    baseline_model = load_model(args, device, training=False)
    ds = CelebA(CELEBA_ROOT, split='train',
                download=False,
                transform=get_default_forget_transform(args.img_size, args.n_bits),
                target_type='attr')
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    male_idx = 20
    n_bins = 2 ** args.n_bits
    # from train import calc_z_shapes
    # z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    # all_images = None
    # with torch.no_grad():
    #     for i in range(100):
    #         cur_zs = []
    #         for shape in z_shapes:
    #             cur_zs.append(torch.randn(128, *shape, device=device) * 0.5)
    #         cur_images = model.reverse(cur_zs, reconstruct=False)
    #         if all_images is None:
    #             all_images = cur_images.cpu()
    #         else:
    #             all_images = torch.cat((all_images, cur_images.cpu()), dim=0)
    # torch.save(all_images.cpu(), save_path)

    out_imgs_data = {}
    min_val, max_val = np.inf, -np.inf
    global_idx = 0
    for i, batch in enumerate(dl):
        x, y = batch
        x = x.to(device) + torch.rand_like(x, device=device) / n_bins
        with torch.no_grad():
            log_p, logdet, _ = baseline_model(x)
            logdet = logdet.mean()
            nll_before = (- log_p - logdet).cpu().numpy()
            cur_distances_before = get_realtive_distance(nll_before, mu, sigma)
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
            nll_after = (- log_p - logdet).cpu().numpy()
            cur_distances_after = get_realtive_distance(nll_after, mu, sigma)
        diff = cur_distances_after - cur_distances_before
        if (diff < 0).any():
            print("Found negative diff")
        if diff.max() > max_val:
            max_val = diff.max()
            max_idx = diff.argmax().item()
            out_imgs_data["max"] = {"gender": "male" if y[max_idx, male_idx] == 1.0 else "female",
                                    "before": nll_before[max_idx].item(),
                                    "after": nll_after[max_idx].item(),
                                    "diff": diff[max_idx].item(),
                                    "filename": ds.filename[global_idx + max_idx]}
        if diff.min() < min_val:
            min_val = diff.min()
            min_idx = diff.argmin().item()
            out_imgs_data["min"] = {"gender": "male" if y[max_idx, male_idx] == 1.0 else "female",
                                    "before": nll_before[min_idx].item(),
                                    "after": nll_after[min_idx].item(),
                                    "diff": diff[min_idx].item(),
                                    "filename": ds.filename[global_idx + min_idx]}
        print("min: ", min_val)
        print("max: ", max_val)
        if i % 10 == 0:
            save_dict_as_json(out_imgs_data, f"{base_dir}/out_data.json")
        print(f"finished {i + 1}/{len(dl)}")
        global_idx += x.size(0)
    save_dict_as_json(out_imgs_data, f"{base_dir}/out_data.json")


def distribution2trace(dist: Union[str, np.ndarray], fig: go.Figure, convert2bpd=False, **kwargs):
    if isinstance(dist, str):
        dist = torch.load(dist).numpy()
    if convert2bpd:
        M = 128 * 128 * 3
        n_bins = 2 ** 5
        dist = (dist + (M * math.log(n_bins))) / (math.log(2) * M)
    mu, sigma = dist.mean(), dist.std()
    n_points = int(math.sqrt(dist.size))
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, n_points)
    y = np_gaussian_pdf(x, mu, sigma)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', **kwargs))
    return mu, sigma


def ablation_plot_distributions(dist_files: List[str],
                                names: List[str],
                                save_path: str,
                                save_scores: bool = True,
                                convert2bpd: bool = False):
    plotly_init()
    fig = go.Figure(layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    x_title = 'BPD' if convert2bpd else 'NLL'
    fig.update_xaxes(showline=False, linecolor='blue', title=x_title)
    fig.update_yaxes(showline=False, linecolor='red', title='Density')
    line_width = 2
    # first add baseline
    colors = ["rgb(79,50,178)", "rgb(139,120,59)", "rgb(0,156,142)"]
    if len(dist_files) > len(colors):
        colors = plotly.colors.qualitative.Plotly
    baseline_dist_path = "models/baseline/continue_celeba/distribution_stats/valid_partial_10000/nll_distribution.pt"
    baseline_mu, baseline_sigma = distribution2trace(baseline_dist_path, fig, convert2bpd=convert2bpd,
                                                     # name='base',
                                                     name=names[0],
                                                     line=dict(color='black', width=line_width + 1, dash='dash'))
    out = {"baseline": {"mu": baseline_mu.astype(float), "sigma": baseline_sigma.astype(float)}}
    for i, (dist_file, name) in enumerate(zip(dist_files, names)):
        # text_name = re.match(r"(.*)_\d\d_image_id_.*", name).group(1)
        # text_name = text_name.replace("_", "-")
        # if 'forward' in text_name:
        #     text_name = "no-reverse"
        # elif 'backward' in text_name:
        #     text_name = "no-forward"
        text_name = name
        mu, sigma = distribution2trace(dist_file, fig, convert2bpd=convert2bpd, name=text_name,
                                       line=dict(color=colors[i], width=line_width))
        cur_forward = forward_kl_univariate_gaussians(baseline_mu, baseline_sigma, mu, sigma)
        cur_backward = reverse_kl_univariate_gaussians(baseline_mu, baseline_sigma, mu, sigma)
        out[text_name] = {
            "forward_kl": cur_forward.astype(float),
            "backward_kl": cur_backward.astype(float),
            "mu": mu.astype(float),
            "sigma": sigma.astype(float)}
    set_fig_config(fig)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1),
        xaxis=dict(range=[0, 3]))
    # if zoom:
    #     fig.update_layout(xaxis=dict(range=[0, 1.5]))
    # x_ticks = [i for i in range(0, 31, 5)]
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals=x_ticks,
    #         ticktext=[str(x) for x in x_ticks]))
    if save_path.endswith("html"):
        fig.write_html(save_path)
    else:
        fig.write_image(save_path)
    if save_scores:
        save_dict_as_json(out, f"scores.json")


def supp_normality_gaussians(valid=True):
    name = "valid" if valid else "train"
    dist_path = "models/baseline/continue_celeba/distribution_stats/"
    dist_path += f"{name}_partial_10000/nll_distribution.pt"
    dist = torch.load(dist_path).numpy()
    mu, sigma = dist.mean(), dist.std()
    plotly_init()
    fig = go.Figure(layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    set_fig_config(fig)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showline=False, gridcolor='blue', title='NLL', showgrid=True)
    fig.update_yaxes(showline=False, gridcolor='red', title='Density', showgrid=True)
    line_width = 2
    n_points = int(math.sqrt(dist.size))
    # x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, n_points)
    x = np.linspace(dist.min(), dist.max(), n_points)
    y = np_gaussian_pdf(x, mu, sigma)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=line_width, color='black')))
    fig.add_trace(go.Histogram(x=dist, histnorm='probability density', nbinsx=n_points, opacity=0.5, marker_color='#9A04C7'))
    fig.write_image(f"images/supp/normality_gaussians_{name}.pdf")


class VideoArgs(NamedTuple):
    n_row: int = 4


class CopyImagesArgs(NamedTuple):
    start_idx: int = 8
    end_idx: int = 16


class LatexGridArgs(NamedTuple):
    folder_name: str
    start_idx: int = 8
    end_idx: int = 16


def same_identities_images(exp_name: str,
                           images_indices: List[Union[str, int]],
                           sampling_steps: List[Union[str, int]],
                           output_dir: str,
                           copy_images_args: Optional[CopyImagesArgs] = None,
                           video_args: Optional[VideoArgs] = None,
                           latex_table: Optional[LatexGridArgs] = None):
    assert isinstance(images_indices, list) and images_indices
    if isinstance(images_indices[0], int):
        images_indices = [str(idx) for idx in images_indices]
    assert isinstance(sampling_steps, list) and sampling_steps
    if isinstance(sampling_steps[0], int):
        sampling_steps = [str(idx) for idx in sampling_steps]
    os.makedirs(output_dir, exist_ok=True)
    if copy_images_args is not None:
        for step in sampling_steps:
            for idx in images_indices[copy_images_args.start_idx: copy_images_args.end_idx]:
                src_image = f"{exp_name}/images/{step}/temp_5_sample_{idx}.png"
                dest = f"{output_dir}/step_{step}_id_{idx}.png"
                shutil.copy(src_image, dest)
    if video_args is not None:
        dir_list = [d for d in os.listdir(f"{exp_name}/images") if os.path.isdir(f"{exp_name}/images/{d}")]
        out_path = f"{output_dir}/vid.mp4"
        dir_list.sort(key=lambda dir_name: int(dir_name))
        pil_to_tens = ToTensor()
        vid_images = []
        for d in dir_list:
            cur_images = os.listdir(f"{exp_name}/images/{d}")
            cur_images = list(filter(lambda name: name.replace("temp_5_sample_", "").replace(".png", "") in images_indices, cur_images))
            cur_images.sort(key=lambda name: int(name.replace("temp_5_sample_", "").replace(".png", "")))
            cur_images = torch.stack([pil_to_tens(Image.open(f"{exp_name}/images/{d}/{im}")) for im in cur_images])
            cur_grid_image = make_grid(cur_images, nrow=video_args.n_row).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
                "cpu", torch.uint8).numpy()
            vid_images.append(cur_grid_image)
        images2video(vid_images, video_path=out_path, fps=25)
    if latex_table is not None:
        identities = images_indices[latex_table.start_idx:latex_table.end_idx]
        with open(f"{output_dir}/table.tex", "w") as out_file:
            for line_idx, idx in enumerate(identities):
                cur_str = '\t'
                for step_idx, step in enumerate(sampling_steps):
                    # if step_idx == 0:
                    #     cur_str += r'\raisebox{\wr}{\rotatebox{90}{' + step + '}}&'
                    cur_str += r'\includegraphics[frame,width=\ww,keepaspectratio]{supp/images/' + latex_table.folder_name + '/step_' + step + '_id_' + idx + '.png}'
                    if step_idx != len(sampling_steps) - 1:
                        cur_str += r'&'
                if line_idx != len(identities) - 1:
                    cur_str += r'\tabularnewline'
                print(cur_str)
                out_file.write(cur_str + '\n')


def same_identity_different_attributes(exps_names: List[str],
                                       images_indices: List[Union[str, int]],
                                       sampling_steps: List[Union[str, int]],
                                       output_dir: str,
                                       copy_images_args: Optional[CopyImagesArgs] = None,
                                       video_args: Optional[VideoArgs] = None,
                                       latex_table: Optional[LatexGridArgs] = None):
    assert isinstance(images_indices, list) and images_indices
    if isinstance(images_indices[0], int):
        images_indices = [str(idx) for idx in images_indices]
    assert isinstance(sampling_steps, list) and sampling_steps
    if isinstance(sampling_steps[0], int):
        sampling_steps = [str(idx) for idx in sampling_steps]
    os.makedirs(output_dir, exist_ok=True)
    if copy_images_args is not None:
        for exp_name in exps_names:
            cur_name = exp_name.split("/")[-1].lower()
            for step in sampling_steps:
                for idx in images_indices[copy_images_args.start_idx: copy_images_args.end_idx]:
                    src_image = f"{exp_name}/images/{step}/temp_5_sample_{idx}.png"
                    dest = f"{output_dir}/{cur_name}_step_{step}_id_{idx}.png"
                    shutil.copy(src_image, dest)
    if video_args is not None:
        # each column will hold images of the same attribute change. Each row will hold images of the same identity
        assert video_args.n_row == (len(exps_names) + 1)
        relevant_identities = images_indices[:video_args.n_row]
        pil_to_tens = ToTensor()
        out_path = f"{output_dir}/vid.mp4"
        vid_images = []
        n_samples = sorted([d for d in os.listdir(f"{exps_names[0]}/images") if os.path.isdir(f"{exps_names[0]}/images/{d}")],
                           key=lambda dir_name: int(dir_name))
        draw_names = [exp_name.split("/")[-1].lower().replace("forget", "- ").replace("increase", "+ ").replace("_", "").replace("mouthslightlyopen", "open mouth") for exp_name in exps_names]
        draw_names.insert(len(exps_names) // 2, "original")
        font = font_manager.FontProperties(family='serif')
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, size=17)
        for n in n_samples[:-50]:
            cur_frame_images = []
            for identity in relevant_identities:
                for col_idx, exp_name in enumerate(exps_names):
                    if len(exps_names) // 2 - 1 < col_idx <= len(exps_names) // 2:
                        cur_frame_images.append(f"{exp_name}/images/0/temp_5_sample_{identity}.png")
                    cur_frame_images.append(f"{exp_name}/images/{n}/temp_5_sample_{identity}.png")
            cur_frame_images = torch.stack([pil_to_tens(Image.open(im)) for im in cur_frame_images])
            cur_frame_grid = make_grid(cur_frame_images, nrow=video_args.n_row).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            grid_h, grid_w, grid_c = cur_frame_grid.shape
            text_area_size = 100
            with_text = np.zeros((grid_h + text_area_size, grid_w, grid_c), dtype=np.uint8)
            with_text[text_area_size:] = cur_frame_grid
            with_text = Image.fromarray(with_text)
            draw = ImageDraw.Draw(with_text)
            offset = 0
            for i, text in enumerate(draw_names):
                if i == len(draw_names) // 2:
                    offset = 30
                draw.text((grid_w // len(draw_names) * i + offset, text_area_size // 2), text, (255, 255, 255), font=font)

            vid_images.append(np.asarray(with_text))
        images2video(vid_images, video_path=out_path, fps=20)

    if latex_table is not None:
        identity_idx = images_indices[0]
        with open(f"{output_dir}/table.tex", "w") as out_file:
            for exp_idx, exp_name in enumerate(exps_names):
                cur_name = exp_name.split("/")[-1].lower()
                cur_str = '\t'
                for step_idx, step in enumerate(sampling_steps):
                    # if step_idx == 0:
                    #     cur_str += r'\raisebox{\wr}{\rotatebox{90}{' + exp_name + '}}&'
                    cur_str += r'\includegraphics[frame,width=\ww,keepaspectratio]{supp/images/' + latex_table.folder_name + f'/{cur_name}_step_' + step + '_id_' + identity_idx + '.png}'
                    if step_idx != len(sampling_steps) - 1:
                        cur_str += r'&'
                if exp_idx != len(exps_names) - 1:
                    cur_str += r'\tabularnewline'
                print(cur_str)
                out_file.write(cur_str + '\n')


if __name__ == '__main__':
    set_all_seeds(37)
    logging.getLogger().setLevel(logging.INFO)
    base_dir = "experiments/forget_attributes_thresh_4"
    # exp_name = "experiments/forget_attributes_thresh_4/forget_Blond_Hair"
    # images_indices = [5, 11, 17, 20, 34, 37, 41, 68, 87, 93, 95, 98, 115, 122, 123, 125]
    # images_indices = [87, 93, 95, 98, 115, 122, 123, 125]
    sampling_steps = [0, 25, 50, 75, 100, 125, 150, 175]
    # image_indices = [4, 10, 15, 21, 22, 25, 30, 35]
    # image_indices = [10, 15, 21, 25, 30, 35]
    out_dir = "outputs/supp_grids/same_identity_change_attributes"
    exps_names = [base_dir + "/" + s for s in ['forget_No_Beard', 'forget_Mouth_Slightly_Open', 'increase_Eyeglasses', 'increase_Bald']]
    # copy_image_args = CopyImagesArgs(start_idx=0, end_idx=5)
    images_indices_increase_blond = [2, 7, 14, 16, 22, 26, 29, 30, 71, 74, 77, 54, 56, 59, 60, 91]
    images_indices_increase_male = [8, 12, 6, 4]
    add_increase_blond_indices = [17]
    same_identities_images(exp_name=base_dir + '/increase_blond_smiling',
                           images_indices=add_increase_blond_indices,
                           sampling_steps=sampling_steps[:-1],
                           output_dir="outputs/supp_grids/increase_blond_smiling",
                           copy_images_args=CopyImagesArgs(start_idx=0, end_idx=16),
                           # copy_images_args=CopyImagesArgs(start_idx=0, end_idx=8),
                           # video_args=VideoArgs(),
                           video_args=None,
                           latex_table=LatexGridArgs(folder_name='increase_blond_smiling', start_idx=0, end_idx=16))

    # video_args = VideoArgs(n_row=5)
    # latex_args = LatexGridArgs(folder_name="same_identity_change_attributes")
    # images_args = CopyImagesArgs(start_idx=1, end_idx=2)
    # same_identity_different_attributes(exps_names, [image_indices[1]], sampling_steps, out_dir, copy_images_args=None, video_args=None, latex_table=latex_args)
    # same_identities_images(exp_name,
    #                        images_indices,
    #                        sampling_steps,
    #                        out_dir,
    #                        copy_images_args=None,
    #                        video_args=VideoArgs(),
    #                        latex_table=LatexGridArgs(folder_name='same_identity_change_forget_blond'))
    # ids = TEST_IDENTITIES[:1]
    # exp_prefix = "experiments/ablation"
    # exp_suffix = "distribution_stats/valid_partial_10000/nll_distribution.pt"
    # ablation_relevant_names = ["backward_only", "forward_only"]
    # dist_files = [f"{exp_prefix}/{name}_15_image_id_10015/{exp_suffix}" for name in ablation_relevant_names]
    # dist_files.append("experiments/forget_all_identities_log_10/15_image_id_10015/distribution_stats/valid_partial_10000/nll_distribution.pt")
    # names = ["                   "] * 3
    # base_dir = "experiments/forget_attributes_2/debias_male_2"
    # base_dir = "experiments/forget_attributes_2/debias_male_3"
    # save_name = "debias.pdf"
    # male_female_teaser()
    # plot_teaser_figure(file_name='teaser_forget.pdf')
    # supp_normality_gaussians(valid=True)
    # supp_normality_gaussians(valid=False)
    # ablation_plot_distributions(dist_files, names, save_path="images/supp/ablation.pdf", save_scores=False, convert2bpd=True)
    # find_teaser_figure_image(base_dir=base_dir)