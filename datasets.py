import os
import pdb
from typing import Union, Optional, List, Callable
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import pandas as pd
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torch
import torchvision.datasets as vision_datasets
import random


class ForgetSampler(Sampler):
    """
    Sampler that enables equals probability to all samples in cases of len(dataset) < batch_size.
    """
    def __init__(self, data_source, batch_size):
        assert len(data_source) < batch_size, f"This sampler is used only when len(data_source) < batch_size but" \
                                              f" got batch_size={batch_size}, len(data_source)={len(data_source)}"
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.reps = batch_size // len(data_source)
        self.sequential = [i % len(self.data_source) for i in range(len(self.data_source) * self.reps)]

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        remainder_part = torch.randint(low=0, high=len(self.data_source),
                                       size=(self.batch_size - len(self.sequential), )).tolist()
        indices = self.sequential + remainder_part
        random.shuffle(indices)
        return iter(indices)


class CelebAPartial(vision_datasets.CelebA):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 exclude_images: List[str] = None,
                 exclude_identities: List[int] = None,
                 exclude_indices: List[int] = None,
                 include_only_images: List[str] = None,
                 include_only_identities: List[int] = None,
                 include_only_indices: List[int] = None):
        super().__init__(root, split=split, target_type=target_type, transform=transform,
                         target_transform=target_transform, download=download)
        """
        Added arguments on top of CelebA dataset from torchvision, in order to exclude certain files according to 
        given args. expecting (possible) args, or including only certain files:
        1. exclude_images: List[str] - list of file_names (with original name from celeba) to exclude
        2. exclude_identities: List[int] - list of identities to exclude (from possible {1, 2, ..., 10177} identities)
        3. exclude_indices: List[int] - list of indices to exclude
        4. include_only_images: List[str] - list of file_names (with original name from celeba) to include in dataset
        5. include_only_identities: List[int] - list of identities to include in dataset
        6. include_only_indices: List[int] - list of indices to include in dataset
        """
        assert not ((exclude_images or exclude_identities) and (include_only_images or include_only_identities)), \
            "excluding and including are mutually exclusive"
        all_exclude_indices = []
        if exclude_images is not None:
            all_exclude_indices += self.__images2idx(exclude_images)
        if exclude_identities is not None:
            all_exclude_indices += self.__identities2idx(exclude_identities)
        if exclude_indices is not None:
            all_exclude_indices += exclude_indices

        if all_exclude_indices:
            self.filename = [self.filename[i] for i in range(len(self.filename)) if i not in all_exclude_indices]
            index_tensor = torch.ones(len(self.attr), dtype=bool)
            index_tensor[all_exclude_indices] = False
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

        include_indices = []
        if include_only_images is not None:
            include_indices += self.__images2idx(include_only_images)
        if include_only_identities is not None:
            include_indices += self.__identities2idx(include_only_identities)
        if include_only_indices is not None:
            include_indices += include_only_indices

        if include_indices:
            self.filename = [self.filename[i] for i in include_indices]
            index_tensor = torch.zeros(len(self.attr), dtype=bool)
            index_tensor[include_indices] = True
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

    def __images2idx(self, images_names) -> List[int]:
        res = []
        for path in images_names:
            assert path in self.filename, f"{path} is not in the dataset"
            res.append(self.filename.index(path))
        return res

    def __identities2idx(self, identities) -> List[int]:
        assert 'identity' in self.target_type, "identity is not in the target_type"
        res = []
        for identity in identities:
            assert 1 <= identity <= 10177, f"{identity} is not in the dataset"
            cur_indices = ((self.identity == identity).nonzero(as_tuple=True)[0]).tolist()
            res += cur_indices
        return res


class PathsDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


class RandomDataset(Dataset):
    def __init__(self, img_size, num_images, transform=None, uniform=False, clip=False):
        self.img_size = img_size
        self.num_images = num_images
        self.transform = transform
        self.uniform = uniform
        self.clip = clip

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.uniform:
            img = torch.rand(*self.img_size)
        else:
            img = torch.randn(*self.img_size)
        if self.transform:
            img = self.transform(img)
        if self.clip:
            img = torch.clamp(img, -0.5, 0.5)

        return img, 0


class Cub2011Dataset(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Cub2011Dataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = torch.tensor(sample.target - 1)  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
