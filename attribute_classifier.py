import json
import os
import logging
from easydict import EasyDict
from torch.utils.data import DataLoader
from utils import CELEBA_ROOT, load_resnet_for_binary_cls, \
    get_resnet_50_normalization, load_model, save_dict_as_json, BASELINE_MODEL_PATH
import pytorch_lightning as pl
import torch
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Compose, Resize
from torchmetrics import Accuracy, F1Score, AUROC, ROC, MetricCollection
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Glow
from train import calc_z_shapes
from constants import CELEBA_NUM_ATTRIBUTES, CELEBA_ATTRIBUTES_MAP, CELEBA_TRAIN_SIZE

ATTRIBUTES_INDICES_PATH = "attribute_classifier/attributes_indices.json"
DEFAULT_CLS_CKPT = "attribute_classifier/checkpoints/train_all_attr_cls/epoch=6-step=8904.ckpt"


def compute_celeba_attribute_count(out_path: str = 'models/attribute_classifier/all_attributes/train_counts.txt',
                                   split: str = 'train'):
    transform = get_cls_default_transform()
    ds = CelebA(root=CELEBA_ROOT, split=split, transform=transform, target_type="attr",
                target_transform=lambda x: x.float())
    dl = DataLoader(ds, batch_size=2048, num_workers=16, drop_last=False)
    counts = torch.zeros(CELEBA_NUM_ATTRIBUTES)
    for x, y in dl:
        counts += y.sum(dim=0)
    with open(out_path, 'w') as f:
        f.write('\n'.join([str(int(w.item())) for w in counts]))


class CelebaAttributeCls(pl.LightningModule):
    def __init__(self, pretrained_backbone=True,
                 attr_counts_path: str = 'models/attribute_classifier/all_attributes/train_counts.txt'):
        super().__init__()
        self.model = load_resnet_for_binary_cls(pretrained=pretrained_backbone, num_outputs=CELEBA_NUM_ATTRIBUTES)
        with open(attr_counts_path, "r") as count_f:
            attr_counts = [int(line.strip()) for line in count_f.readlines()]
        self.bceloss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(CELEBA_TRAIN_SIZE - att) / CELEBA_TRAIN_SIZE for att in attr_counts]))
        self.train_metrics = torch.nn.ModuleDict({CELEBA_ATTRIBUTES_MAP[i]:
                                                  MetricCollection([Accuracy(), F1Score()]).clone(
                prefix=f"_train_{CELEBA_ATTRIBUTES_MAP[i]}")
            for i in range(CELEBA_NUM_ATTRIBUTES)})
        self.val_metrics = torch.nn.ModuleDict({CELEBA_ATTRIBUTES_MAP[i]:
                                                MetricCollection([Accuracy(), F1Score()]).clone(
                prefix=f"_val_{CELEBA_ATTRIBUTES_MAP[i]}")
            for i in range(CELEBA_NUM_ATTRIBUTES)})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.bceloss(y_hat, y)
        y = y.int()
        # loggings and accuracy calculations
        self.log('train_loss', loss)
        for i in range(CELEBA_NUM_ATTRIBUTES):
            cur_output = self.train_metrics[CELEBA_ATTRIBUTES_MAP[i]](y_hat[:, i], y[:, i])
            self.log_dict(cur_output)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.bceloss(y_hat, y)
        y = y.int()
        # loggings and accuracy calculations
        self.log('val_loss', loss)
        for i in range(CELEBA_NUM_ATTRIBUTES):
            cur_output = self.val_metrics[CELEBA_ATTRIBUTES_MAP[i]](y_hat[:, i], y[:, i])
            self.log_dict(cur_output)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.__log_metrics_epoch_end(self.val_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_epoch_end(self, outputs) -> None:
        self.__log_metrics_epoch_end(self.train_metrics)

    def __log_metrics_epoch_end(self, metrics: torch.nn.ModuleDict):
        cur_metrics = {CELEBA_ATTRIBUTES_MAP[i]:
                           metrics[CELEBA_ATTRIBUTES_MAP[i]].compute() for i in range(CELEBA_NUM_ATTRIBUTES)}
        # log and reset metrics
        for i in range(CELEBA_NUM_ATTRIBUTES):
            self.log_dict({f"epoch_{k}": v for k, v in cur_metrics[CELEBA_ATTRIBUTES_MAP[i]].items()})
            metrics[CELEBA_ATTRIBUTES_MAP[i]].reset()


def get_dataset(split='train'):
    transform = get_cls_default_transform()
    ds = CelebA(root=CELEBA_ROOT, split=split, transform=transform, target_type="attr",
                target_transform=lambda x: x.float())
    return ds


def load_classifier(ckpt_path='', device=None):
    classifier = CelebaAttributeCls.load_from_checkpoint(ckpt_path)
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


def evaluate_model(model: CelebaAttributeCls, dl: DataLoader, device=None, save_dir=None):
    metrics = {i: {'accuracy': Accuracy(),
                   'f1': F1Score(),
                   'auc': AUROC(),
                   'roc': ROC(pos_label=1)} for i in range(CELEBA_NUM_ATTRIBUTES)}
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in dl:
            x, y = batch
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            y_hat = model(x)
            y = y.int()
            total += y.shape[0]
            for i in range(CELEBA_NUM_ATTRIBUTES):
                cur_metrics = metrics[i]
                for metric in cur_metrics:
                    cur_metrics[metric](y_hat[:, i].detach().cpu(), y[:, i].detach().cpu())

    results = {i: {} for i in range(CELEBA_NUM_ATTRIBUTES)}
    for i in range(CELEBA_NUM_ATTRIBUTES):
        cur_metrics = metrics[i]
        for metric in cur_metrics:
            if metric == 'roc':
                continue
            results[i][metric] = cur_metrics[metric].compute().item()

    res = {CELEBA_ATTRIBUTES_MAP[i]: results[i] for i in range(CELEBA_NUM_ATTRIBUTES)}
    res["Total Test Examples"] = total
    if save_dir:
        for i in range(CELEBA_NUM_ATTRIBUTES):
            fpr, tpr, _ = metrics[i]['roc'].compute()
            auc = results[i]['auc']
            plot_roc(fpr, tpr, auc, os.path.join(save_dir, f"{CELEBA_ATTRIBUTES_MAP[i]}_roc.png"))
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(res, f, indent=4)
    return res


def train():
    exp_name = "train_all_attr_cls"
    logger = WandbLogger(project='celeba-atribute-classifier', save_dir='attribute_classifier/', name=exp_name)
    model = CelebaAttributeCls(pretrained_backbone=True)
    batch_size = 128
    train_ds = get_dataset(split='train')
    val_ds = get_dataset(split='valid')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    os.makedirs(f'attribute_classifier/checkpoints/{exp_name}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'attribute_classifier/checkpoints/{exp_name}',
                                          save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(logger=logger, max_epochs=15, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
    trainer.fit(model, train_dl, val_dl)


def analyze_glow_attributes(n_samples, save_path, ckpt_path=BASELINE_MODEL_PATH, device=None):
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
        logging.info(f"{i + 1}/{n_iter}")
    data = {"total": total,
            "positive males": positive_males,
            "positive glasses": positive_glasses,
            "males ratio": positive_males / total,
            "glasses ratio": positive_glasses / total}
    save_dict_as_json(data, save_path)


def get_cls_default_transform(img_size=128) -> Compose:
    return Compose([Resize((img_size, img_size)), ToTensor(), get_resnet_50_normalization()])


def compute_celeba_attribute_indices(split='train',
                                     out_dir: str = 'models/attribute_classifier/all_attributes/indices'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ds = CelebA(root=CELEBA_ROOT, split=split, target_type='attr', download=False, transform=ToTensor())
    indices = {i: [] for i in range(CELEBA_NUM_ATTRIBUTES)}
    dl = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=16)
    count = 0
    for i, (x, y) in enumerate(dl):
        for j in range(CELEBA_NUM_ATTRIBUTES):
            cur_unnormalized = (y[:, j] == 1).nonzero().squeeze()
            normalized_indices = cur_unnormalized + count
            indices[j].extend(normalized_indices.tolist())
        logging.info(f"{i + 1}/{len(dl)}")
        count += x.shape[0]
    for v in indices.values():
        assert len(v) == len(set(v))
    with open("models/attribute_classifier/all_attributes/train_counts.txt", "r") as count_file:
        counts = [int(line.strip()) for line in count_file.readlines()]
    for i in range(CELEBA_NUM_ATTRIBUTES):
        assert len(indices[i]) == counts[i], f"idx : {i} has different length between indices: {len(indices[i])} " \
                                             f"and total count: {counts[i]}"
        torch.save(torch.tensor(indices[i], dtype=torch.long), f"{out_dir}/{i}.pt")


def load_indices_cache(attr_idx):
    indices = torch.load(f"models/attribute_classifier/all_attributes/indices/{attr_idx}.pt")
    return indices


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    pass
