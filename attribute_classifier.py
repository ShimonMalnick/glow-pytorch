import json
import os
from time import time
import logging
from typing import Iterator, Union
import wandb
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import CelebAPartial
from utils import CELEBA_ROOT, load_resnet_for_binary_cls, CELEBA_MALE_ATTR_IDX, CELEBA_GLASSES_ATTR_IDX, \
    get_resnet_50_normalization, load_model, save_dict_as_json, get_args, get_partial_dataset
import pytorch_lightning as pl
import torch
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip
from torchmetrics import Accuracy, F1Score, AUROC, ROC
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Glow
from train import calc_z_shapes
from forget import make_forget_exp_dir, get_data_iterator, get_default_forget_transform, get_eval_data, \
    calc_regular_loss, compute_dataloader_bpd, plot_bpd_histograms, save_model_optimizer


# Constants #
# adjustment for the loss on imbalanced dataset
CELEBA_MALE_FRACTION = (202599 - 84434) / 84434
CELEBA_GLASSES_FRACTION = (202599 - 13193) / 13193
GLASSES_IDX = 0
MALE_IDX = 1
GLOW_BASELINE_CKPT = "models/baseline/continue_celeba/model_090001.pt"
ATTRIBUTES_INDICES_PATH = "attribute_classifier/attributes_indices.json"


class TwoAttributesClassifier(pl.LightningModule):
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.model = load_resnet_for_binary_cls(pretrained=pretrained_backbone, num_outputs=2)
        self.bceloss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CELEBA_GLASSES_FRACTION,
                                                                           CELEBA_MALE_FRACTION]))
        self.glasses_accuracy_train = Accuracy()
        self.male_accuracy_train = Accuracy()
        self.glasses_accuracy_val = Accuracy()
        self.male_accuracy_val = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.bceloss(y_hat, y)
        y = y.int()
        # loggings and accuracy calculations
        glasses_acc = self.glasses_accuracy_train(y_hat[:, GLASSES_IDX], y[:, GLASSES_IDX])
        male_acc = self.male_accuracy_train(y_hat[:, MALE_IDX], y[:, MALE_IDX])
        self.log('train_loss', loss)
        self.log('train_glass_accuracy', glasses_acc)
        self.log('train_male_accuracy', male_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.bceloss(y_hat, y)
        y = y.int()
        # loggings and accuracy calculations
        glasses_acc = self.glasses_accuracy_val(y_hat[:, GLASSES_IDX], y[:, GLASSES_IDX])
        male_acc = self.male_accuracy_val(y_hat[:, MALE_IDX], y[:, MALE_IDX])
        self.log('val_loss', loss)
        self.log("val_glasses_accuracy", glasses_acc)
        self.log("val_male_accuracy", male_acc)

        return loss

    def validation_epoch_end(self, outputs) -> None:
        glasses_accuracy = self.glasses_accuracy_val.compute()
        male_accuracy = self.male_accuracy_val.compute()
        # log metrics
        self.log("epoch_val_glasses_accuracy", glasses_accuracy)
        self.log("epoch_val_male_accuracy", male_accuracy)
        # reset all metrics
        self.glasses_accuracy_val.reset()
        self.male_accuracy_val.reset()
        print(f"\nval glasses accuracy: {glasses_accuracy:.4}, "
              f"val male accuracy: {male_accuracy:.4}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_epoch_end(self, outputs) -> None:
        glasses_accuracy = self.glasses_accuracy_train.compute()
        male_accuracy = self.male_accuracy_train.compute()
        # log metrics
        self.log("epoch_train_glasses_accuracy", glasses_accuracy)
        self.log("epoch_train_male_accuracy", male_accuracy)
        # reset all metrics
        self.glasses_accuracy_train.reset()
        self.male_accuracy_train.reset()
        print(f"\ntraining glasses accuracy: {glasses_accuracy:.4}, "
              f"training male accuracy: {male_accuracy:.4}")


def get_dataset(split='train'):
    attrs_indices = [CELEBA_GLASSES_ATTR_IDX, CELEBA_MALE_ATTR_IDX]
    transform = Compose([Resize((128, 128)), RandomHorizontalFlip(), ToTensor(), get_resnet_50_normalization()])
    ds = CelebA(root=CELEBA_ROOT, split=split, transform=transform, target_type="attr",
                target_transform=lambda x: x[attrs_indices].float())
    return ds


def load_classifier(ckpt_path='models/attribute_classifier/epoch=5-step=7632.ckpt', device=None):
    classifier = TwoAttributesClassifier.load_from_checkpoint(ckpt_path)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def plot_roc(fpr, tpr, auc, save_path):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f'AUC: {auc:.4}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(save_path)


def evaluate_model(model: TwoAttributesClassifier, dataset: DataLoader, device=None, save_dir=None):
    NAME_IDX = 0
    METRIC_IDX = 1
    glasses_metrics = [('accuracy', Accuracy()), ('f1', F1Score()), ('auc', AUROC()), ('roc', ROC(pos_label=1))]
    male_metrics = [('accuracy', Accuracy()), ('f1', F1Score()), ('auc', AUROC()), ('roc', ROC(pos_label=1))]
    num_metrics = len(male_metrics)
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in dataset:
            x, y = batch
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            y_hat = model(x)
            y = y.int()
            total += y.shape[0]
            for i in range(num_metrics):
                glasses_metrics[i][METRIC_IDX](y_hat[:, GLASSES_IDX].detach().cpu(), y[:, GLASSES_IDX].detach().cpu())
                male_metrics[i][METRIC_IDX](y_hat[:, MALE_IDX].detach().cpu(), y[:, MALE_IDX].detach().cpu())

    glasses_results = {}
    male_results = {}
    for i in range(num_metrics - 1):  # without roc metrics
        glasses_results[glasses_metrics[i][NAME_IDX]] = glasses_metrics[i][METRIC_IDX].compute().item()
        male_results[male_metrics[i][NAME_IDX]] = male_metrics[i][METRIC_IDX].compute().item()

    res = {"Glasses": glasses_results, "Male": male_results, "Total Test Examples": total}
    if save_dir:
        fpr, tpr, _ = glasses_metrics[num_metrics - 1][METRIC_IDX].compute()
        plot_roc(fpr.numpy(), tpr.numpy(), glasses_results['auc'], f"{save_dir}/glasses_roc.png")
        fpr, tpr, _ = male_metrics[num_metrics - 1][METRIC_IDX].compute()
        plot_roc(fpr, tpr, male_results['auc'], f"{save_dir}/male_roc.png")
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(res, f, indent=4)


def train():
    exp_name = "train"
    logger = WandbLogger(project='celeba-atribute-classifier', save_dir='attribute_classifier/', name=exp_name)
    model = TwoAttributesClassifier(pretrained_backbone=True)
    batch_size = 128
    train_ds = get_dataset(split='train')
    val_ds = get_dataset(split='valid')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    os.mkdir(f'attribute_classifier/checkpoints/{exp_name}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'attribute_classifier/checkpoints/{exp_name}',
                                          save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(logger=logger, max_epochs=10, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
    trainer.fit(model, train_dl, val_dl)


def analyze_glow_attributes(n_samples, save_path, ckpt_path=GLOW_BASELINE_CKPT, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {"ckpt_path": ckpt_path,
            "n_flow": 32,
            "n_block": 4,
            "affine": False,
            "no_lu": False,
            "batch": 256,
            "temp": 0.7}
    args = EasyDict(args)
    glow: Glow = load_model(args, device, training=False)
    cls = load_classifier(device=device)
    cls_norm = get_resnet_50_normalization()

    z_shapes = calc_z_shapes(3, 128, args.n_flow, args.n_block)
    n_iter = int(n_samples / args.batch)
    total = 0
    positive_males = 0
    positive_glasses = 0
    for i in range(n_iter):
        cur_zs = []
        for shape in z_shapes:
            cur_zs.append(torch.randn(args.batch, *shape).to(device) * args.temp)
        with torch.no_grad():
            images = glow.reverse(cur_zs, reconstruct=False)
            out = cls(cls_norm(images))
            total += out.shape[0]
            positive_glasses += torch.sum(out[:, GLASSES_IDX] >= 0.5).item()
            positive_males += torch.sum(out[:, MALE_IDX] >= 0.5).item()
        logging.info(f"{i + 1}/{n_iter}")
    data = {"total": total,
            "positive males": positive_males,
            "positive glasses": positive_glasses,
            "males ratio": positive_males / total,
            "glasses ratio": positive_glasses / total}
    save_dict_as_json(data, save_path)


def get_celeba_attributes_indices(split='train', save_dir='attribute_classifier'):
    ds = get_dataset(split=split)
    data = {'male': [], 'glasses': []}
    start = time()
    for i in range(len(ds)):
        _, y = ds[i]
        if (i + 1) % 1000 == 0:
            print(f"finished {i} images in {round(time() - start, 2)} seconds")
            start = time()
        if y[GLASSES_IDX] == 1:
            data['glasses'].append(i)
        if y[MALE_IDX] == 1:
            data['male'].append(i)
    save_dict_as_json(data, f"{save_dir}/attributes_indices.json")


def get_attribute_ds(args, transform, include_attribute=True) -> CelebAPartial:
    with open(ATTRIBUTES_INDICES_PATH, "r") as f:
        attributes_indices = json.load(f)
    assert args.forget_attribute in ['male', 'glasses']
    indices = attributes_indices[args.forget_attribute]
    attr_index = CELEBA_GLASSES_ATTR_IDX if args.forget_attribute == 'glasses' else CELEBA_MALE_ATTR_IDX

    def target_transform(x):
        return x[attr_index].float()

    params = {"transform": transform,
              'target_type': 'attr',
              "target_transform": target_transform}

    if include_attribute:
        params["include_only_indices"] = indices
    else:
        params["exclude_indices"] = indices

    ds = get_partial_dataset(**params)
    return ds


def get_attribute_data_iter(args, transform, include_attribute=True) -> Iterator:
    ds = get_attribute_ds(args, transform, include_attribute)
    data_iterator = get_data_iterator(ds, args.batch, args.num_workers)
    return data_iterator


def forget_alpha(args, remember_iter: Iterator, forget_iter: Iterator, model: Union[Glow, torch.nn.DataParallel],
                 device, optimizer: torch.optim.Optimizer, eval_dl: DataLoader):
    cpu_device = torch.device('cpu')
    cls = load_classifier(device=cpu_device)
    cls_norm = get_resnet_50_normalization()
    z_shapes = calc_z_shapes(3, 128, args.n_flow, args.n_block)
    attribute_index = MALE_IDX if args.forget_attribute == 'male' else GLASSES_IDX
    assert args.alpha is None or 0.0 <= args.alpha <= 1.0
    n_bins = 2 ** args.n_bits
    for i in range(args.iter):
        log_dict = {}
        forget_batch = next(forget_iter)[0].to(device)
        forget_p, forget_det, _ = model(forget_batch + torch.rand_like(forget_batch) / n_bins)
        forget_det = forget_det.mean()
        forget_loss, forget_p, forget_det = calc_regular_loss(forget_p, forget_det, args.img_size, n_bins)
        if not args.debias:
            forget_loss.mul_(-1.0)

        if args.alpha < 1.0:
            remember_batch = next(remember_iter)[0].to(device)
            remember_p, remember_det, _ = model(remember_batch + torch.rand_like(remember_batch) / n_bins)
            remember_det = remember_det.mean()
            remember_loss, remember_p, remember_det = calc_regular_loss(remember_p, remember_det, args.img_size, n_bins)
            log_dict["remember"] = {"loss": remember_loss.item(),
                                    "log_p": remember_p.item(),
                                    "log_det": remember_det.item()}
            loss = args.alpha * forget_loss + (1 - args.alpha) * remember_loss
        else:
            loss = forget_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.log_every > 0 and i % args.log_every == 0:
            eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model, device, eval_dl,
                                              reduce=False).cpu()
            plot_bpd_histograms(i + 1, args.exp_name, eval_bpd=eval_bpd.numpy())

        positive = 0
        for n_sample in range(4):
            cur_zs = []
            for shape in z_shapes:
                cur_zs.append(torch.randn(256, *shape).to(device) * 0.7)
            with torch.no_grad():
                images = model.module.reverse(cur_zs, reconstruct=False).to(cpu_device)
                out = cls(cls_norm(images))
                positive += torch.sum(out[:, attribute_index] >= 0.5).item()
        log_dict.update({"forget": {"loss": forget_loss.item(),
                                    "log_p": forget_p.item(),
                                    "log_det": forget_det.item()},
                         "positive": positive})
        wandb.log(log_dict)

        logging.info(f"Iter: {i + 1} positive: {positive} positive ratio: {(positive / (256 * 4)):.3f}")

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model, optimizer, save_optim=False)
    if args.save_every is not None:
        save_model_optimizer(args, 0, model, optimizer, last=True, save_optim=False)


def forget_attribute():
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True, forget_attribute=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="forget_attribute")
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = get_attribute_ds(args, transform, include_attribute=True)
    forget_iter = get_data_iterator(forget_ds, args.batch, args.num_workers)
    args["forget_ds_len"] = len(forget_ds)
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.forget_lr)
    remember_ds = get_attribute_ds(args, transform, include_attribute=False)
    remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
    args["remember_ds_len"] = len(remember_ds)
    eval_batches, eval_dl = get_eval_data(args, remember_ds, device)
    wandb.init(project="Forget Attribute", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    forget_alpha(args, remember_iter, forget_iter, model, device, forget_optimizer, eval_dl)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    # forget_attribute()
    base = "experiments/forget_attribute"
    dirs = ['male_debias_alpha_1_save', 'male_debias_alpha_0.7_save', 'glasses_debias_alpha_1',
                 'glasses_debias_alpha_0.7']
    # dirs = ['male_debias_alpha_1_save']
    device = torch.device("cuda")
    for d in dirs:
        dir_path = base + "/" + d
        ckpt_path = dir_path + "/checkpoints/model_last.pt"
        args = {"ckpt_path": ckpt_path,
                "n_flow": 32,
                "n_block": 4,
                "affine": False,
                "no_lu": False,
                "batch": 64,
                "temp": 0.7}
        args = EasyDict(args)
        glow: Glow = load_model(args, device, training=False)
        z_shapes = calc_z_shapes(3, 128, args.n_flow, args.n_block)
        cur_zs = []
        for shape in z_shapes:
            cur_zs.append(torch.randn(args.batch, *shape).to(device) * args.temp)
        with torch.no_grad():
            images = glow.reverse(cur_zs, reconstruct=False).cpu()
        save_image(images + 0.5, f'{dir_path}/samples.png', nrow=8)
