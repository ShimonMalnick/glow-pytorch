import json
import os
import pdb
from typing import List
import plotly
import wandb
from forget_attribute import save_model_images, forget_attribute, compute_attribute_step_stats, get_model_latents, \
    image_folders_to_grid_video
from torch.utils.data import Subset
from easydict import EasyDict
from cifar_classifier import get_dataset, get_cifar_classes_cache
import torch
from forget import make_forget_exp_dir
from utils import set_all_seeds, get_args, load_model, save_dict_as_json, plotly_init, save_fig, CIFAR_GLOW_CKPT_PATH, \
    get_default_forget_transform
import logging
from model import Glow
from train import calc_z_shapes
import plotly.graph_objects as go
from constants import CIFAR10_CLASS2IDX, CIFAR10_IDX2CLASS


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True, forget_attribute=True, cifar=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    dir_name = "forget_cifar_class" if not args.dir_name else args.dir_name
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name=dir_name)
    os.makedirs(f"experiments/{args.exp_name}/random_samples")
    os.makedirs(f"experiments/{args.exp_name}/cls")
    logging.info(args)
    output_device = torch.device(f"cuda:{train_devices[0]}")
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices,
                                              output_device=output_device)
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)

    n_classes = args.cifar_n_classes
    assert 0 <= args.forget_attribute < n_classes, "Must specify a class to forget"
    class2indices = get_cifar_classes_cache(train=True)
    forget_indices = class2indices[args.forget_attribute]
    cifar_ds = get_dataset(train=True, transform=transform)
    forget_ds = Subset(cifar_ds, forget_indices)
    remember_ds = Subset(cifar_ds, list(set(range(len(cifar_ds))) - set(forget_indices)))

    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    wandb.init(project="forget_cifar_class", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    compute_attribute_step_stats(args, 0, model, None, remember_ds, forget_ds, train_devices[0], init=True)
    logging.info("Starting forget attribute procedure")
    generate_samples = False
    latents = get_model_latents(args.img_size, args.n_flow, args.n_block, args.n_sample, args.temp, output_device)
    if args.n_sample:
        save_model_images(model, f'experiments/{args.exp_name}/images/0', args.n_sample, latents, temp=args.temp)
    forget_attribute(args, remember_ds, forget_ds, model,
                     original_model, train_devices, output_device, original_model_device,
                     forget_optimizer, generate_samples=generate_samples, const_latents=latents, cls_type='cifar')
    image_folders_to_grid_video(f'experiments/{args.exp_name}', n_images=16, nrow=4)


def plot_multiple_classes(dirs: List[str], attribute_indices: List[int],
                          save_path='forget_attributes_multiple_attributes.pdf'):
    assert len(dirs) == len(attribute_indices)
    for d in dirs:
        if os.path.exists(f"{d}/cls_results.json"):
            continue
        iter_names = os.listdir(f"{d}/cls")
        out = {}
        for name in iter_names:
            cur_iter_name = name.replace(".json", "")
            with open(f"{d}/cls/{name}") as cur_j:
                cur_results = json.load(cur_j)
            out[cur_iter_name] = cur_results
        save_dict_as_json(out, f"{d}/cls_results.json")
    plotly_init()
    fig = go.Figure()
    files = [f"{d}/cls_results.json" for d in dirs]
    # keys = [i - 1 if i != 0 else i for i in range(0, 110, 10)]
    colors = plotly.colors.qualitative.D3_r
    for i, file in enumerate(files):
        cur_name = CIFAR10_IDX2CLASS[attribute_indices[i]]
        with open(file, "r") as f:
            results = json.load(f)
        results = {int(k): v[str(attribute_indices[i])] / sum(v.values()) for k, v in sorted(results.items(), key=lambda item: int(item[0]))}
        results = {k: v for k, v in results.items() if k <= 150}
        fig.add_trace(go.Scatter(x=list(results.keys()),
                                 y=[round(v * 100, 2) for k, v in results.items()],
                                 name=cur_name.replace("_", " "), line=dict(color=colors[i])))
    fig.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False, gridcolor='blue', title_text="Step", showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showgrid=False, gridcolor='red', title_text="Classified samples [%]", showline=True, linewidth=2,
                     linecolor='black')
    # fig.write_html(f"forget_attributes_multiple_attributes.html")
    fig.update_layout(width=500, height=250,
                      font=dict(family="Serif", size=14),
                      margin_l=5, margin_t=5, margin_b=5, margin_r=5)
    save_fig(fig, save_path)


@torch.no_grad()
def save_images(n_samples=16, args=None, ckpt_path=CIFAR_GLOW_CKPT_PATH):
    if args is None:
        base_dir = "experiments/train/train_cifar"
        with open(f"{base_dir}/args.json") as input_j:
            args = EasyDict(json.load(input_j))
    args.ckpt_path = ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, training=False, device=device)

    z_shapes = calc_z_shapes(3, 32, 48, 3)
    temp = args.temp
    cur_zs = []
    for shape in z_shapes:
        cur_zs.append(torch.randn(n_samples, *shape).to(device) * temp)
    os.makedirs(f"{base_dir}/images")
    save_model_images(model, f"{base_dir}/images", n_samples, temp=temp, latents=cur_zs)


if __name__ == '__main__':
    set_all_seeds(seed=37)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    main()
