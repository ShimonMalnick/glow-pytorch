import os
import logging
from typing import Union, Optional
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset
from utils import load_resnet_for_binary_cls, get_resnet_50_normalization, load_model, save_dict_as_json, CIFAR_ROOT, \
    CIFAR_GLOW_CKPT_PATH, CIFAR_CLS_CKPT_PATH
import pytorch_lightning as pl
import torch
from torchvision.transforms import ToTensor, Compose, Resize
from torchmetrics import Accuracy, F1Score, AUROC, ROC
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Glow
from train import calc_z_shapes
import torchvision.datasets as vision_datasets


class Cifar10Cls(pl.LightningModule):
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.model = load_resnet_for_binary_cls(pretrained=pretrained_backbone, num_outputs=10)
        self.celoss = torch.nn.CrossEntropyLoss()
        self.train_metrics = torch.nn.ModuleDict({'accuracy': Accuracy(task='multiclass', num_classes=10),
                                                  'f1_score': F1Score(task='multiclass', num_classes=10)})
        self.val_metrics = torch.nn.ModuleDict({'accuracy': Accuracy(task='multiclass', num_classes=10),
                                                'f1_score': F1Score(task='multiclass', num_classes=10)})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.celoss(y_hat, y)
        # loggings and accuracy calculations
        self.log('train_loss', loss)
        self.train_metrics['accuracy'].update(y_hat, y)
        self.train_metrics['f1_score'].update(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.celoss(y_hat, y)
        # loggings and accuracy calculations
        self.log('val_loss', loss)
        self.val_metrics['accuracy'].update(y_hat, y)
        self.val_metrics['f1_score'].update(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.__log_metrics_epoch_end(self.val_metrics, 'val_')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_epoch_end(self, outputs) -> None:
        self.__log_metrics_epoch_end(self.train_metrics, 'train_')

    def __log_metrics_epoch_end(self, metrics: torch.nn.ModuleDict, name_prefix=""):
        for k in metrics:
            self.log(f"{name_prefix}epoch_{k}", metrics[k].compute())


def get_dataset(train=True):
    transform = get_cls_default_transform()
    ds = vision_datasets.CIFAR10(CIFAR_ROOT, transform=transform, download=True, train=train)

    return ds


def load_cifar_classifier(ckpt_path=CIFAR_GLOW_CKPT_PATH, device=None):
    classifier = Cifar10Cls.load_from_checkpoint(ckpt_path)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def plot_roc(fpr, tpr, auc, save_path):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f'AUC: {auc:.4}' if auc else '')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(save_path)


@torch.no_grad()
def evaluate_model(model: Cifar10Cls, dl: DataLoader, device=None, save_dir=None, plot_roc_curve=True):
    metrics = {'accuracy': Accuracy(task='multiclass', num_classes=10),
               'f1': F1Score(task='multiclass', num_classes=10),
               'auc': AUROC(task='multiclass', num_classes=10),
               'roc': ROC(task='multiclass', num_classes=10)}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    total = 0
    for batch in dl:
        x, y = batch
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        y_hat = model(x)
        y = y.int()
        total += y.shape[0]
        for name in metrics:
            cur_metric = metrics[name]
            cur_metric(y_hat.detach().cpu(), y.detach().cpu())

    results = {}
    for name in metrics:
        if name == 'roc':
            continue
        cur_metric = metrics[name]
        results[name] = cur_metric.compute().item()

    results["Total Test Examples"] = total

    if save_dir:
        fpr, tpr, _ = metrics['roc'].compute()
        for i in range(10):
            plot_roc(fpr[i], tpr[i], auc=None, save_path=os.path.join(save_dir, f"roc_class_{i}.png"))
        save_dict_as_json(results, os.path.join(save_dir, "metrics.json"))
    return results


def train():
    base_dir = 'cifar_classifier'
    exp_name = "train_cifar10_cls"
    logger = WandbLogger(project='cifar-classifier', save_dir=base_dir, name=exp_name)
    model = Cifar10Cls(pretrained_backbone=True)
    batch_size = 128
    train_ds = get_dataset(train=True)
    val_ds = get_dataset(train=True)
    n = len(train_ds)
    rand_indices = torch.randperm(n)
    train_ds = Subset(train_ds, rand_indices[:int(0.9 * n)])
    val_ds = Subset(val_ds, rand_indices[int(0.9 * n):])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    os.makedirs(f'{base_dir}/checkpoints/{exp_name}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'{base_dir}/checkpoints/{exp_name}',
                                          save_top_k=2, monitor="val_epoch_accuracy", mode='max')
    trainer = pl.Trainer(logger=logger, max_epochs=15, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
    trainer.fit(model, train_dl, val_dl)
    return model


@torch.no_grad()
def analyze_glow_on_cifar(n_samples, save_path, ckpt_path=CIFAR_GLOW_CKPT_PATH, device=None):
    args, device, glow = get_model_args_device(ckpt_path, device)
    cls = load_cifar_classifier(ckpt_path=CIFAR_CLS_CKPT_PATH, device=device)
    cls_norm = get_resnet_50_normalization()

    z_shapes = calc_z_shapes(3, 128, args.n_flow, args.n_block)
    n_iter = int(n_samples / args.batch)
    total = 0
    classes_count = {i: 0 for i in range(10)}
    for i in range(n_iter):
        cur_zs = []
        for shape in z_shapes:
            cur_zs.append(torch.randn(args.batch, *shape).to(device) * args.temp)
        images = glow.reverse(cur_zs, reconstruct=False)
        out = cls(cls_norm(images))
        total += out.shape[0]
        out = out.argmax(dim=1)
        for j in range(out.shape[0]):
            classes_count[out[j].item()] += 1
        logging.info(f"{i + 1}/{n_iter}")
    percentage = {i: classes_count[i] / total for i in range(10)}

    classes_count['total'] = total
    classes_count['percentage'] = percentage
    save_dict_as_json(classes_count, save_path)


def get_model_args_device(ckpt_path, device):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {"ckpt_path": ckpt_path,
            "n_flow": 48,
            "n_block": 3,
            "affine": False,
            "no_lu": False,
            "batch": 512,
            "temp": 0.7}
    args = EasyDict(args)
    glow: Glow = load_model(args, device, training=False)
    return args, device, glow


@torch.no_grad()
def evaluate_cifar_random_samples_classification(samples: Union[str, torch.Tensor],
                                           out_path: str,
                                           device: Optional[torch.device] = None,
                                           cls=None,
                                           cls_transform=None):
    if isinstance(samples, str):
        samples = torch.load(samples)
    elif not isinstance(samples, torch.Tensor):
        raise ValueError("samples_file should be either str or torch.Tensor")
    if cls is None:
        cls = load_cifar_classifier(device=device)
    if cls_transform is None:
        cls_transform = get_resnet_50_normalization()

    classes_count = {i: 0 for i in range(10)}
    samples = cls_transform(samples + 0.5)
    samples = samples.to(device)
    y_hat = cls(samples)
    pred = torch.argmax(y_hat, dim=-1)
    for j in range(pred.shape[0]):
        classes_count[pred[j].item()] += 1
    if out_path:
        save_dict_as_json(classes_count, out_path)
    return classes_count


def get_cls_default_transform(img_size=128) -> Compose:
    return Compose([Resize((img_size, img_size)), ToTensor(), get_resnet_50_normalization()])


def get_cifar_classes_cache(train=True):
    train_str = "train" if train else "test"
    cache_path = f'cifar_classifier/class_caches/{train_str}.dict'
    if not os.path.isfile(cache_path):
        ds = get_dataset(train=train)
        out = {i: [] for i in range(10)}
        for i in range(len(ds)):
            cur_label = ds[i][1]
            out[cur_label].append(i)
        torch.save(out, cache_path)
    return torch.load(cache_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    train()
