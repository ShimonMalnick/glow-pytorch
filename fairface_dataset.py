import logging
import os
from typing import Optional, Callable, Dict, List
import torch
from torch.utils.data import Dataset, Subset
import csv
from PIL import Image

if os.environ['FAIRFACE_DIR'] is None:
    LOCAL_FAIRFACE_ROOT = "../datasets/fairface-img-margin025-trainval"
else:
    LOCAL_FAIRFACE_ROOT = os.environ['FAIRFACE_DIR']
LABELS_NAME2IDX = {
    "age": {'0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8},

    "gender": {'Male': 0,
               'Female': 1},

    "race": {'White': 0,
             'Black': 1,
             'Latino_Hispanic': 2,
             'East Asian': 3,
             'Southeast Asian': 4,
             'Indian': 5,
             'Middle Eastern': 6}}

LABELS_IDX2NAME = {k: {v: sub_key for sub_key, v in LABELS_NAME2IDX[k].items()} for k in LABELS_NAME2IDX.keys()}

LABELS_POSITIONS2NAME = {0: 'age', 1: 'gender', 2: 'race'}
LABELS_NAME2POSITION = {v: k for k, v in LABELS_POSITIONS2NAME.items()}


def label2name(label_type: str, index: int):
    return LABELS_IDX2NAME[label_type][index]


def universal_label2name(label: int) -> str:
    if label < 7:
        return label2name('race', label)
    elif label < 9:
        return label2name('gender', label - 7)
    elif label < 18:
        return label2name('age', label - 9)
    else:
        raise ValueError(f"Label {label} is out of range, should be in [0, 17]")


def default_label_transform(label: List) -> torch.Tensor:
    return torch.tensor(label)


def get_one_hot_labels(label: List[int]) -> torch.Tensor:
    out = torch.zeros(18)
    race = label[LABELS_NAME2POSITION['race']]
    out[race] = 1.0
    gender = label[LABELS_NAME2POSITION['gender']]
    out[gender + 7] = 1.0
    age = label[LABELS_NAME2POSITION['age']]
    out[age + 9] = 1.0
    return out


def one_type_label_wrapper(label_type: str, one_hot=False) -> Callable:
    shape = len(LABELS_NAME2IDX[label_type])
    shift = 0 + 7 * (label_type == 'gender') + 9 * (label_type == 'age')

    def one_hot_wrapper(label: List[int]) -> torch.Tensor:
        out = torch.zeros(shape)
        pos = label[LABELS_NAME2POSITION[label_type]]
        out[pos + shift] = 1.0
        return out

    def single_wrapper(label: List[int]) -> torch.Tensor:
        return torch.tensor(label[LABELS_NAME2POSITION[label_type]])

    return one_hot_wrapper if one_hot else single_wrapper


class FairFaceDataset(Dataset):
    def __init__(
            self,
            root: str = LOCAL_FAIRFACE_ROOT,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = get_one_hot_labels,
            data_type: str = 'train'
    ):
        assert data_type in ['train', 'val'], "data_type must be either 'train' or 'val'"
        self.root = root
        self.transform = transform
        self.target_trasform = target_transform
        self.data_type = data_type
        self.labels = self.__load_labels()
        self.images = os.listdir(os.path.join(root, data_type))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            self.images = self.images[:100]
            self.labels = {key: self.labels[key] for key in self.images}
        assert len(self.images) == len(self.labels), "inconsistency between labels and images"

    def __load_labels(self) -> Dict[str, List]:
        with open(os.path.join(self.root, f"fairface_label_{self.data_type}.csv")) as csv_file:
            data = list(csv.reader(csv_file, delimiter=",", skipinitialspace=True))[1:]  # remove headers

        raw_data = [row[:-1] for row in data]  # remove last label - irrelevant for us
        data = []
        for j, row in enumerate(raw_data):
            name, labels = row[0], row[1:]
            mapped_row = [name.lstrip(self.data_type + "/")]
            for i in range(len(labels)):
                cur_label_name = LABELS_POSITIONS2NAME[i]
                cur_label_idx = LABELS_NAME2IDX[cur_label_name][labels[i]]
                mapped_row.append(cur_label_idx)
            data.append(mapped_row)
        return {row[0]: row[1:] for row in data}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cur_image_path = self.images[idx]
        cur_image = Image.open(os.path.join(self.root, self.data_type, cur_image_path))
        if self.transform:
            cur_image = self.transform(cur_image)

        cur_label = self.labels[cur_image_path]
        logging.debug(f"image: {cur_image_path} raw label {[LABELS_IDX2NAME['age'][cur_label[0]], LABELS_IDX2NAME['gender'][cur_label[1]], LABELS_IDX2NAME['race'][cur_label[2]]]}")
        if self.target_trasform:
            cur_label = self.target_trasform(cur_label)
        return cur_image, cur_label


def compute_label_dataset_indices(save_dir, data_type='train'):
    os.makedirs(save_dir, exist_ok=True)
    dataset = FairFaceDataset(data_type=data_type)
    indices = {label: [] for label in range(18)}
    for idx in range(len(dataset)):
        cur_label = dataset[idx][1]
        for i in range(18):
            if cur_label[i] == 1:
                indices[i].append(idx)
    for label in indices:
        torch.save(torch.tensor(indices[label]), os.path.join(save_dir, f"{label}.pt"))


def get_label_indices(index, data_type='train') -> List[int]:
    indices = torch.load(os.path.join("../etc"
                                      "/fairface_indices", data_type, f"{index}.pt"))
    return indices


def get_label_dataset(index, data_type='train') -> Dataset:
    ds = FairFaceDataset(data_type=data_type)
    indices = get_label_indices(index, data_type=data_type)
    subset_ds = Subset(ds, indices)
    return subset_ds

