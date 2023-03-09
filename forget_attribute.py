import json
import os
import re
from glob import glob
from typing import Union, List, Dict, Optional, Callable
from pdb import set_trace
import numpy as np
import plotly
import wandb
from PIL import Image
from torch.utils.data import Subset, Dataset, DataLoader
from easydict import EasyDict
from torchvision.transforms import ToTensor

from attribute_classifier import CelebaAttributeCls, DEFAULT_CLS_CKPT, load_indices_cache, get_cls_default_transform
import torch
from forget import make_forget_exp_dir, get_data_iterator, get_kl_loss_fn, get_log_p_parameters, \
    get_kl_and_remember_loss, prob2bpd
from utils import set_all_seeds, get_args, get_default_forget_transform, load_model, CELEBA_ROOT, \
    save_dict_as_json, compute_dataloader_bpd, save_model_optimizer, BASE_MODEL_PATH, get_resnet_50_normalization, \
    plotly_init, save_fig, set_fig_config, images2video
import logging
from model import Glow
from torchvision.datasets import CelebA
from train import calc_z_shapes
from constants import CELEBA_ATTRIBUTES_MAP
import plotly.graph_objects as go
from torchvision.utils import make_grid, save_image


def load_cls(ckpt_path: str = DEFAULT_CLS_CKPT, device=None) -> CelebaAttributeCls:
    model = CelebaAttributeCls.load_from_checkpoint(ckpt_path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def compute_attribute_step_stats(args: EasyDict,
                                 step: int,
                                 model: torch.nn.DataParallel,
                                 device, remember_ds: Dataset,
                                 forget_ds: Dataset,
                                 sampling_device: torch.device,
                                 init: bool = False,
                                 generate_samples: bool = True) -> torch.Tensor:
    model.eval()
    cur_forget_indices = torch.randperm(len(forget_ds))[:1024]
    cur_forget_ds = Subset(forget_ds, cur_forget_indices)
    cur_forget_dl = DataLoader(cur_forget_ds, batch_size=256, num_workers=args.num_workers)

    cur_remember_indices = torch.randperm(len(remember_ds))[:1024]
    cur_remember_ds = Subset(remember_ds, cur_remember_indices)
    cur_remember_dl = DataLoader(cur_remember_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    if (step + 1) % args.log_every == 0 or init:
        eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model,
                                          device, cur_remember_dl, reduce=False).cpu()
        args.eval_mu = eval_bpd.mean().item()
        args.eval_std = eval_bpd.std().item()

        forget_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model,
                                            device, cur_forget_dl, reduce=False).cpu()

        logging.info(
            f"eval_mu: {args.eval_mu}, eval_std: {args.eval_std}, forget_bpd: {forget_bpd.mean().item()} for iteration {step}")
        forget_signed_distance = (forget_bpd.mean().item() - args.eval_mu) / args.eval_std
        wandb.log({f"eval_bpd": eval_bpd.mean().item(),
                   "eval_mu": args.eval_mu,
                   "eval_std": args.eval_std,
                   "forget_distance": forget_signed_distance},
                  commit=False)
        # if generate_samples:
            # to generate images using the reverse function, we need the module itself from the DataParallel wrapper
            # generate_random_samples(model.module, sampling_device, args,
            #                         f"experiments/{args.exp_name}/random_samples/step_{step}.pt")
        return forget_bpd

    return torch.tensor([0], dtype=torch.float)


def forget_attribute(args: EasyDict, remember_ds: Dataset, forget_ds: Dataset,
                     model: Union[Glow, torch.nn.DataParallel],
                     original_model: Glow,
                     training_devices: List[torch.device],
                     output_device: torch.device,
                     original_model_device: torch.device,
                     optimizer: torch.optim.Optimizer,
                     generate_samples: bool = True,
                     const_latents=None):
    kl_loss_fn = get_kl_loss_fn(args.loss)
    sigmoid = torch.nn.Sigmoid()
    cls = load_cls(device=original_model_device)
    cls_transform = get_resnet_50_normalization()
    main_device = torch.device(f"cuda:{training_devices[0]}")
    n_bins = 2 ** args.n_bits
    n_pixels = args.img_size * args.img_size * 3

    remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
    forget_iter = get_data_iterator(forget_ds, args.batch, args.num_workers)
    for i in range(args.iter):
        cur_forget_images = next(forget_iter)[0].to(main_device)
        log_p, logdet, _ = model(cur_forget_images + torch.rand_like(cur_forget_images) / n_bins)
        logdet = logdet.mean()
        cur_bpd = prob2bpd(log_p + logdet, n_bins, n_pixels)
        signed_distance = (cur_bpd - (args.eval_mu + args.eval_std * (args.forget_thresh)))
        forget_loss = sigmoid(signed_distance ** 2).mean()

        remember_batch = next(remember_iter)[0].to(main_device)
        remember_batch += torch.rand_like(remember_batch) / n_bins
        with torch.no_grad():
            orig_p, orig_det, _ = original_model(remember_batch.to(original_model_device))
            orig_dist = orig_p + orig_det.mean()
            orig_mean, orig_std = get_log_p_parameters(n_bins, n_pixels, orig_dist, device=main_device)

        kl_loss, remember_loss = get_kl_and_remember_loss(args, kl_loss_fn, model, n_bins, n_pixels, orig_mean,
                                                          orig_std, remember_batch)

        loss = args.alpha * forget_loss + (1 - args.alpha) * remember_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.n_sample:
            # for debugging runs without saving images
            save_model_images(model, f'experiments/{args.exp_name}/images/{i + 1}', args.n_sample, const_latents, temp=args.temp)
        cur_forget_bpd = compute_attribute_step_stats(args, i, model, main_device, remember_ds, forget_ds,
                                                      training_devices[0], generate_samples=generate_samples)
        wandb.log({"forget": {"loss": forget_loss.item(), "kl_loss": kl_loss.item()},
                   "remember": {"loss": remember_loss.item()},
                   "forget_bpd_mean": cur_forget_bpd.mean().item()
                   })

        logging.info(
            f"Iter: {i + 1} Forget Loss: {forget_loss.item():.5f}; Remember Loss: {remember_loss.item():.5f}")

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model.module, optimizer, save_optim=False)
        if (i == 0) or (i + 1) % args.cls_every == 0:
            cur_latents = get_model_latents(args.img_size, args.n_flow, args.n_block, args.cls_n_samples, args.temp, output_device)
            with torch.no_grad():
                bs = 256
                assert not args.cls_n_samples % bs, f"args.cls_n_samples needs to be divisible by {bs}"
                cur_images = None
                for idx in range(args.cls_n_samples // bs):
                    cur_batch = [latent[bs * idx:bs * (idx + 1)] for latent in cur_latents]
                    if not idx:
                        cur_images = model.module.reverse(cur_batch, reconstruct=False)
                    else:
                        cur_images = torch.cat([cur_images, model.module.reverse(cur_batch, reconstruct=False)])
            evaluate_random_samples_classification(cur_images.to(original_model_device), f"experiments/{args.exp_name}/cls/{i+1}.json",
                                                   device=original_model_device, cls=cls, cls_transform=cls_transform, sigmoid=sigmoid)

    if args.save_every is not None:
        save_model_optimizer(args, 0, model.module, optimizer, last=True, save_optim=False)

    return model


def save_random_samples(exp_dir: str, model_ckpt: str, out_dir: str, num_samples=4, with_baseline=True):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(exp_dir, "args.json"), "r") as f:
        args = EasyDict(json.load(f))
    args.ckpt_path = model_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, training=False)
    models = [("", model)]
    if with_baseline:
        args.ckpt_path = BASE_MODEL_PATH
        models.append(("baseline_", load_model(args, device, training=False)))
    z_shapes = calc_z_shapes(3, 128, 32, 4)
    temp = 0.5
    for prefix, cur_model in models:
        cur_zs = []
        for shape in z_shapes:
            cur_zs.append(torch.randn(num_samples, *shape).to(device) * temp)
        with torch.no_grad():
            model_images = cur_model.reverse(cur_zs, reconstruct=False).cpu() + 0.5  # added 0.5 for normalization
            model_images = model_images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            for i in range(num_samples):
                im = Image.fromarray(model_images[i])
                im.save(os.path.join(out_dir, f"{prefix}temp_{int(temp * 10)}_sample_{i}.png"))


def generate_random_samples(model, device, args, save_path, temp=0.5, n_samples=512, batch_size=256):
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    n_iter = n_samples // batch_size
    all_images = None
    with torch.no_grad():
        for i in range(n_iter):
            cur_zs = []
            for shape in z_shapes:
                cur_zs.append(torch.randn(batch_size, *shape, device=device) * temp)
            cur_images = model.reverse(cur_zs, reconstruct=False)
            if all_images is None:
                all_images = cur_images.cpu()
            else:
                all_images = torch.cat((all_images, cur_images.cpu()), dim=0)
    if save_path:
        torch.save(all_images.cpu(), save_path)
    return all_images


@torch.no_grad()
def evaluate_random_samples_classification(samples: Union[str, torch.Tensor],
                                           out_path: str,
                                           device: Optional[torch.device] = None,
                                           combine_attributes=None,
                                           cls=None,
                                           cls_transform=None,
                                           sigmoid=None):
    if isinstance(samples, str):
        samples = torch.load(samples)
    elif not isinstance(samples, torch.Tensor):
        raise ValueError("samples_file should be either str or torch.Tensor")
    if sigmoid is None:
        sigmoid = torch.nn.Sigmoid()
    if cls is None:
        cls = load_cls(device=device)
    if cls_transform is None:
        cls_transform = get_resnet_50_normalization()
    samples = cls_transform(samples + 0.5)
    samples = samples.to(device)
    y_hat = sigmoid(cls(samples))

    pred = torch.sum(y_hat > 0.5, dim=0)
    out = {}
    if combine_attributes is not None:
        for cur_attributes in combine_attributes:
            key_name = ",".join([CELEBA_ATTRIBUTES_MAP[i] for i in cur_attributes])
            additional_pred = torch.sum(torch.prod(y_hat[:, cur_attributes] > 0.5, dim=-1)).item()
            out[key_name] = {"positive": additional_pred,
                             "fraction": additional_pred / len(samples)}
    for k in CELEBA_ATTRIBUTES_MAP:
        out[CELEBA_ATTRIBUTES_MAP[k]] = {"positive": pred[k].item(), "fraction": pred[k].item() / len(samples)}
    if out_path:
        save_dict_as_json(out, out_path)
    return out


def get_model_latents(img_size, n_flow, n_block, n_sample, temp, device) -> List[torch.Tensor]:
    z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    latents = []
    for shape in z_shapes:
        latents.append(torch.randn(n_sample, *shape, device=device) * temp)
    return latents


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True, forget_attribute=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1] if len(all_devices) > 1 else all_devices
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    dir_name = "forget_attributes" if not args.dir_name else args.dir_name
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name=dir_name)
    os.makedirs(f"experiments/{args.exp_name}/random_samples")
    os.makedirs(f"experiments/{args.exp_name}/cls")
    logging.info(args)
    output_device = torch.device(f"cuda:{train_devices[0]}")
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices, output_device=output_device)
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)

    assert args.forget_attribute is not None, "Must specify attribute to forget"
    forget_indices = load_indices_cache(args.forget_attribute)
    if args.forget_additional_attribute is not None:
        # forget_indices = torch.cat((forget_indices, load_indices_cache(args.forget_additional_attribute)))
        #doing intersection of these groups
        forget_indices = torch.tensor(np.intersect1d(forget_indices, load_indices_cache(args.forget_additional_attribute)))

    celeba_ds = CelebA(root=CELEBA_ROOT, split='train', transform=transform, target_type='attr')
    if 'debias' in args and args.debias == 1:
        remember_ds = Subset(celeba_ds, forget_indices)
        forget_ds = Subset(celeba_ds, list(set(range(len(celeba_ds))) - set(forget_indices.tolist())))
    else:
        forget_ds = Subset(celeba_ds, forget_indices)
        remember_ds = Subset(celeba_ds, list(set(range(len(celeba_ds))) - set(forget_indices.tolist())))

    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    wandb.init(project="forget_attributes_thresh_1", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    compute_attribute_step_stats(args, 0, model, None, remember_ds, forget_ds, train_devices[0], init=True)
    logging.info("Starting forget attribute procedure")
    generate_samples = False
    latents = get_model_latents(args.img_size, args.n_flow, args.n_block, args.n_sample, args.temp, output_device)
    if args.n_sample:
        save_model_images(model, f'experiments/{args.exp_name}/images/0', args.n_sample, latents, temp=args.temp)
    forget_attribute(args, remember_ds, forget_ds, model,
                     original_model, train_devices, output_device, original_model_device,
                     forget_optimizer, generate_samples=generate_samples, const_latents=latents)
    image_folders_to_grid_video(f'experiments/{args.exp_name}', n_images=16, nrow=4)


@torch.no_grad()
def evaluate_experiment_random_samples(exp_dir: str,
                                       device: Optional[torch.device] = None,
                                       save_res=True,
                                       saved_files=False) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_results_path = "models/baseline/continue_celeba/generated_samples/cls.json"
    with open(baseline_results_path, "r") as f:
        baseline_results = json.load(f)
    out_dict = {k: {"baseline": v} for k, v in baseline_results.items()}
    cls = load_cls(device=device)
    models_iterable = []
    if saved_files:
        samples_paths = glob(f"{exp_dir}/random_samples/*.pt")
        models_iterable = [(p, p.split("_")[-1][:-3]) for p in samples_paths]
    else:
        with open(f"{exp_dir}/args.json") as in_args_json:
            args = EasyDict(json.load(in_args_json))
        models_names_raw = os.listdir(f"{exp_dir}/checkpoints")
        model_name_pattern = r"model_(\d+).pt"
        for m in models_names_raw:
            cur_match = re.match(model_name_pattern, m)
            if not cur_match:
                continue
            cur_name = f"{int(cur_match.group(1))}"
            args.ckpt_path = os.path.join(exp_dir, "checkpoints", m)
            cur_model = load_model(args, device, training=False)
            cur_images = generate_random_samples(cur_model, device, args, save_path=None)
            models_iterable.append((cur_images, cur_name))

    os.makedirs(f"{exp_dir}/cls", exist_ok=True)
    for images, name in models_iterable:
        cur_dict = evaluate_random_samples_classification(images, f"{exp_dir}/cls/{name}.json", device=device, cls=cls)
        for k in cur_dict:
            out_dict[k][name] = cur_dict[k]

    out_path = f"{exp_dir}/cls_scores.json"
    if save_res:
        save_dict_as_json(out_dict, out_path)
    return out_dict


def plot_attribute_change(exp_dir: str, attribute_identifier: Union[str, int]):
    if isinstance(attribute_identifier, int):
        attribute_identifier = CELEBA_ATTRIBUTES_MAP[attribute_identifier]
    elif isinstance(attribute_identifier, str):
        assert attribute_identifier in CELEBA_ATTRIBUTES_MAP, f"Unknown attribute {attribute_identifier}"
    else:
        raise ValueError(f"Unknown attribute identifier {attribute_identifier}")
    results_path = f"{exp_dir}/random_samples/cls_scores.json"
    with open(results_path, "r") as results_file:
        results = json.load(results_file)[attribute_identifier]
    results_no_baseline = {int(k): v for k, v in results.items() if k != "baseline"}
    results_no_baseline = {k: v for k, v in sorted(results_no_baseline.items())}
    plotly_init()
    fig = go.Figure(data=go.Scatter(x=list(results_no_baseline.keys()), y=[v["positive"]
                                                                           for v in results_no_baseline.values()]))
    fig.update_layout(showlegend=False, title=f"Attribute {attribute_identifier} positive samples".title(),
                      font=dict(family="Serif", size=16))
    fig.update_xaxes(title_text="Step", showline=True, linewidth=2, linecolor='black', gridcolor='Red')
    fig.update_yaxes(title_text="Classified Samples", showline=True, linewidth=2, linecolor='black', gridcolor='Blue')
    fig.write_html(f"{exp_dir}/random_samples/cls_scores_samples_{attribute_identifier}.html")

    fig = go.Figure(data=go.Scatter(x=list(results_no_baseline.keys()), y=[round(v["fraction"] * 100, 2)
                                                                           for v in results_no_baseline.values()]))
    fig.update_layout(showlegend=False, title=f"Attribute {attribute_identifier} positive samples percentage".title(),
                      font=dict(family="Serif", size=16))
    fig.update_xaxes(title_text="Step", showline=True, linewidth=2, linecolor='black', gridcolor='Red')
    fig.update_yaxes(title_text="Classified Samples [%]", showline=True, linewidth=2, linecolor='black', gridcolor='Blue')
    fig.write_html(f"{exp_dir}/random_samples/cls_scores_percentage_samples_{attribute_identifier}.html")


def plot_multiple_attributes(files: List[str],
                             attribute_indices: List[str],
                             save_path='forget_attributes_multiple_attributes.pdf'):
    assert len(files) == len(attribute_indices)
    plotly_init()
    fig = go.Figure()
    # keys = [i - 1 if i != 0 else i for i in range(0, 110, 10)]
    colors = plotly.colors.qualitative.D3_r
    for i, file in enumerate(files):
        cur_name = CELEBA_ATTRIBUTES_MAP[attribute_indices[i]]
        with open(file, "r") as f:
            results = json.load(f)[cur_name]
        results_no_baseline = {int(k): v for k, v in results.items() if k != "baseline"}
        results_no_baseline = {k: v for k, v in sorted(results_no_baseline.items())}
        keys = [k for k in results_no_baseline.keys() if k <= 150]
        # fig.add_trace(go.Scatter(x=list(results_no_baseline.keys()),
        #                          y=[round(v["fraction"] * 100, 2) for v in results_no_baseline.values()],
        #                          name=cur_name))
        fig.add_trace(go.Scatter(x=keys,
                                 y=[round(v["fraction"] * 100, 2) for k, v in results_no_baseline.items()],
                                 name=cur_name.replace("_", " "), line=dict(color=colors[i])))
    fig.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False, gridcolor='blue', title_text="Step", showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showgrid=False, gridcolor='red', title_text="Classified samples [%]", showline=True, linewidth=2, linecolor='black')
    # fig.write_html(f"forget_attributes_multiple_attributes.html")
    fig.update_layout(width=500, height=250,
                      font=dict(family="Serif", size=14),
                      margin_l=5, margin_t=5, margin_b=5, margin_r=5)
    save_fig(fig, save_path)


@torch.no_grad()
def save_model_images(
        model: Union[torch.nn.DataParallel, Glow],
        out_dir: str,
        n_imgs: int,
        latents: List[torch.Tensor],
        temp: float = 0.5):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    os.makedirs(out_dir, exist_ok=True)
    cur_images = model.reverse(latents, reconstruct=False) + 0.5
    cur_images = cur_images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    for i in range(n_imgs):
        im = Image.fromarray(cur_images[i])
        im.save(os.path.join(out_dir, f"temp_{int(temp * 10)}_sample_{i}.png"))


@torch.no_grad()
def save_images_along_models(exp_dir, n_imgs: int = 64, device=None, temp=0.5, latents=None):
    with open(f"{exp_dir}/args.json") as input_j:
        args = EasyDict(json.load(input_j))
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    models = glob(f"{exp_dir}/checkpoints/model_*.pt")
    os.makedirs(f"{exp_dir}/constant_images", exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if latents is None:
        latents = []
        for shape in z_shapes:
            latents.append(torch.randn(n_imgs, *shape, device=device) * temp)
    for model_p in models:
        model_name = model_p.split("/")[-1][:-3]
        os.makedirs(f"{exp_dir}/{model_name}", exist_ok=True)
        args.ckpt_path = model_p
        model = load_model(args, device, training=False)
        cur_images = model.reverse(latents, reconstruct=False) + 0.5
        cur_images = cur_images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        for i in range(n_imgs):
            im = Image.fromarray(cur_images[i])
            im.save(os.path.join(exp_dir, model_name, f"temp_{int(temp * 10)}_sample_{i}.png"))
        del model


def image_folders_to_grid_video(exp_dir, n_images=16, nrow=4, out_path='', start_idx=0):
    dir_list = os.listdir(f"{exp_dir}/images")
    if not out_path:
        out_path = f"{exp_dir}/vid.mp4"
    dir_list.sort(key=lambda dir_name: int(dir_name))
    pil_to_tens = ToTensor()
    vid_images = []
    for d in dir_list:
        cur_images = os.listdir(f"{exp_dir}/images/{d}")
        cur_images.sort(key=lambda name: int(name.replace("temp_5_sample_", "").replace(".png", "")))
        cur_images = cur_images[start_idx:n_images + start_idx]
        cur_images = torch.stack([pil_to_tens(Image.open(f"{exp_dir}/images/{d}/{im}")) for im in cur_images])
        cur_grid_image = make_grid(cur_images, nrow=nrow).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        vid_images.append(cur_grid_image)
    images2video(vid_images, video_path=out_path, fps=25)


if __name__ == '__main__':
    set_all_seeds(seed=37)
    main()