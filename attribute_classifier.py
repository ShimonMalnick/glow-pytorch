import os
from torch.utils.data import DataLoader
from utils import CELEBA_ROOT, load_resnet_for_binary_cls, CELEBA_MALE_ATTR_IDX, CELEBA_GLASSES_ATTR_IDX, \
    get_resnet_50_normalization
import pytorch_lightning as pl
import torch
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ[
    'TORCH_HOME'] = '/home/yandex/AMNLP2021/malnick/.cache/torch/'  # save resnet50 checkpoint under this directory

# Constants #
# adjustment for the loss on imbalanced dataset
CELEBA_MALE_FRACTION = (202599 - 84434) / 84434
CELEBA_GLASSES_FRACTION = (202599 - 13193) / 13193


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
        glasses_acc = self.glasses_accuracy_train(y_hat[:, 0], y[:, 0])
        male_acc = self.male_accuracy_train(y_hat[:, 1], y[:, 1])
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
        glasses_acc = self.glasses_accuracy_val(y_hat[:, 0], y[:, 0])
        male_acc = self.male_accuracy_val(y_hat[:, 1], y[:, 1])
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


def load_classifier(ckpt_path='attribute_classifier/checkpoints/train/epoch=5-step=7632.ckpt', device=None):
    classifier = TwoAttributesClassifier.load_from_checkpoint(ckpt_path)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def evaluate_model(model: TwoAttributesClassifier, dataset: DataLoader, device=None):
    model.eval()
    with torch.no_grad():
        for batch in dataset:
            x, y = batch
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            y_hat = model(x)
            y = y.int()


if __name__ == '__main__':
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
