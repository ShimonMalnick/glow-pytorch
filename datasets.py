from typing import Union, Optional, List, Callable

from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.datasets as vision_datasets


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
                 include_only_images: List[str] = None,
                 include_only_identities: List[int] = None):
        super().__init__(root, split=split, target_type=target_type, transform=transform,
                         target_transform=target_transform, download=download)
        """
        Added arguments on top of CelebA dataset from torchvision, in order to exclude certain files according to 
        given args. expecting (possible) args, or including only certain files:
        1. exclude_images: List[str] - list of file_names (with original name from celeba) to exclude
        2. exclude_identities: List[int] - list of identities to exclude (from possible {1, 2, ..., 10177} identities)
        3. include_only_images: List[str] - list of file_names (with original name from celeba) to include in dataset
        4. include_only_identities: List[int] - list of identities to include in dataset
        """
        assert not (exclude_images or exclude_identities) and (include_only_images or include_only_identities), \
            "excluding and including are mutually exclusive"
        exclude_indices = []
        if exclude_images is not None:
            exclude_indices += self.__exclude_images(exclude_images)
        if exclude_identities is not None:
            exclude_indices += self.__exclude_identities(exclude_identities)

        if exclude_indices:
            self.filename = [self.filename[i] for i in range(len(self.filename)) if i not in exclude_indices]
            index_tensor = torch.ones(len(self.attr), dtype=bool)
            index_tensor[exclude_indices] = False
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

        include_indices = []
        if include_only_images is not None:
            include_indices += self.__include_only_images(include_only_images)
        if include_only_identities is not None:
            include_indices += self.__include_only_identities(include_only_identities)

        if include_indices:
            self.filename = [self.filename[i] for i in include_indices]
            index_tensor = torch.zeros(len(self.attr), dtype=bool)
            index_tensor[include_indices] = True
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

    def __exclude_images(self, exclude_images) -> List[int]:
        res = []
        for path in exclude_images:
            assert path in self.filename, f"{path} is not in the dataset"
            res.append(self.filename.index(path))
        return res

    def __exclude_identities(self, exclude_identities) -> List[int]:
        assert 'identity' in self.target_type, "identity is not in the target_type"
        res = []
        for identity in exclude_identities:
            assert 1 <= identity <= 10177, f"{identity} is not in the dataset"
            exclude_indices = ((self.identity == identity).nonzero(as_tuple=True)[0]).tolist()
            res += exclude_indices
        return res

    def __include_only_images(self, include_only_images):
        pass

    def __include_only_identities(self, include_only_identities):
        pass


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
