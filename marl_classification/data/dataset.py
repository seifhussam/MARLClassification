# -*- coding: utf-8 -*-
import glob
import pickle as pkl
from os.path import basename, exists, isdir, join, splitext
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import torch as th
import torch.nn.functional as fun
import torchvision.io
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def my_pil_loader(path: str) -> Image.Image:
    """
    "my_pil_loader": loads an image using Pillow, cosnidering the image's path

    Args:
    path (str): image's path

    Return
    Pillow image
    """
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class MNISTDataset(ImageFolder):
    """
    Class of the MNIST dataset, extending the ImageFolder object
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": MNISTDataset class constructor

        Args:
        self (MNISTDataset Object): the MNISTDataset Object itself
        res_path (str): root path
        img_transform (callable): image transformation

        Return:
        None
        """

        mnist_root_path = join(res_path, "downloaded", "mnist_png", "all_png")

        assert exists(mnist_root_path) and isdir(
            mnist_root_path
        ), f"{mnist_root_path} does not exist or is not a directory"

        super().__init__(
            mnist_root_path,
            transform=img_transform,
            target_transform=None,
            loader=my_pil_loader,
            is_valid_file=None,
        )


class RESISC45Dataset(ImageFolder):
    """
    Class of the RESISC45Dataset dataset, extending the ImageFolder object
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": RESISC45Dataset class constructor

        Args:
        self (RESISC45Dataset Object): the RESISC45Dataset Object itself
        res_path (str): root path
        img_transform (callable): image transformation

        Return:
        None
        """

        resisc_root_path = join(res_path, "downloaded", "NWPU-RESISC45")

        assert exists(resisc_root_path) and isdir(
            resisc_root_path
        ), f"{resisc_root_path} does not exist or is not a directory"

        super().__init__(
            resisc_root_path,
            transform=img_transform,
            target_transform=None,
            loader=my_pil_loader,
            is_valid_file=None,
        )


class AIDDataset(ImageFolder):
    """
    Class of the AIDDataset dataset, extending the ImageFolder object
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": AIDDataset class constructor

        Args:
        self (AIDDataset Object): the AIDDataset Object itself
        res_path (str): root path
        img_transform (callable): image transformation

        Return:
        None
        """

        aid_root_path = join(res_path, "downloaded", "AID")

        assert exists(aid_root_path) and isdir(
            aid_root_path
        ), f"{aid_root_path} does not exist or is not a directory"

        super().__init__(
            aid_root_path,
            transform=img_transform,
            target_transform=None,
            loader=my_pil_loader,
            is_valid_file=None,
        )


class KneeMRIDataset(Dataset):
    """
    Class of the KneeMRIDataset dataset, extending the Dataset object
    """

    def __init__(self, res_path: str, _: Callable[[Any], th.Tensor]):
        """
        "__init__": KneeMRIDataset class constructor

        Args:
        self (KneeMRIDataset Object): the KneeMRIDataset Object itself
        res_path (str): root path
        _ (callable): unused

        Return:
        None
        """
        super().__init__()

        self.__knee_mri_root_path = join(res_path, "downloaded", "knee_mri")

        # self.__img_transform = img_transform

        metadata_csv = pd.read_csv(
            join(self.__knee_mri_root_path, "metadata.csv"), sep=","
        )

        tqdm.tqdm.pandas()

        self.__max_depth = -1
        self.__max_width = -1
        self.__max_height = -1
        self.__nb_img = 0

        def __open_pickle_size(fn: str) -> None:
            """
            "__open_pickle_size": uses the Pickle library to open the MRI volume

            Args:
            fn (str): Name of the volume

            Return:
            None
            """
            with open(
                join(self.__knee_mri_root_path, "extracted", fn), "rb"
            ) as f:
                x = pkl.load(f)

            self.__max_depth = max(self.__max_depth, x.shape[0])
            self.__max_width = max(self.__max_width, x.shape[1])
            self.__max_height = max(self.__max_height, x.shape[2])
            self.__nb_img += 1

        metadata_csv["volumeFilename"].progress_map(__open_pickle_size)

        self.__dataset = [
            (str(fn), lbl)
            for fn, lbl in zip(
                metadata_csv["volumeFilename"].tolist(),
                metadata_csv["aclDiagnosis"].tolist(),
            )
        ]

        self.class_to_idx = {
            "healthy": 0,
            "partially injured": 1,
            "completely ruptured": 2,
        }

    def __open_img(self, fn: str) -> th.Tensor:
        """
        "__open_img": used to open an image

        Args:
        fn (str): Name of the volume

        Return:
        torch tensor: opened image
        """
        with open(join(self.__knee_mri_root_path, "extracted", fn), "rb") as f:
            x = pkl.load(f)

        x = th.from_numpy(x).to(th.float)

        # depth
        curr_depth = x.size(0)

        to_pad = self.__max_depth - curr_depth
        pad_1 = to_pad // 2 + to_pad % 2
        pad_2 = to_pad // 2

        # width
        curr_width = x.size(1)

        to_pad = self.__max_width - curr_width
        pad_3 = to_pad // 2 + to_pad % 2
        pad_4 = to_pad // 2

        # height
        curr_height = x.size(2)

        to_pad = self.__max_height - curr_height
        pad_5 = to_pad // 2 + to_pad % 2
        pad_6 = to_pad // 2

        return fun.pad(x, [pad_6, pad_5, pad_4, pad_3, pad_2, pad_1], value=0)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        "__getitem__": gets an item from the KneeMRIDataset

        Args:
        self (KneeMRIDataset): KneeMRIDataset object itself
        index (int): index of the image

        Return:
        Tuple of two torch tensors: image and respective label
        """
        fn = self.__dataset[index][0]

        label = self.__dataset[index][1]
        img = self.__open_img(fn).unsqueeze(0)

        return img, th.tensor(label)

    def __len__(self) -> int:
        """
        "__len__": returns the lenght (number of images) of the dataset

        Args:
        self (KneeMRIDataset): KneeMRIDataset object itself

        Return:
        int: number of images in the dataset
        """
        return self.__nb_img


class WorldStratDataset(Dataset):
    """
    Class of the WorldStratDataset, extending the class Dataset
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ):
        """
        "__init__": WorldStratDataset constructor

        Args:
        self (WorldStratDataset Object): the WorldStratDataset object itself
        res_path (str): path to the folder with the WorldStratDataset
        img_transform (callable): image transformations applied to the dataset

        Return:
        None
        """
        super().__init__()

        self.__root_path = join(res_path, "downloaded", "WorldStrat")

        self.__class_column = "IPCC Class"

        self.__metadata = (
            pd.read_csv(
                join(self.__root_path, "metadata.csv"), sep=",", quotechar='"'
            )
            .dropna()
            .rename(columns={"Unnamed: 0": "folder_name"})
            .groupby("folder_name")[["folder_name", self.__class_column]]
            .first()
        )

        self.__img_transform = img_transform
        self.__img_loader = my_pil_loader

        self.__class_to_idx = {
            c: i
            for i, c in enumerate(
                sorted(self.__metadata[self.__class_column].unique())
            )
        }

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """
        "class_to_idx": creates a dictionary for each class associated with an index

        Args:
        self (WorldStratDataset Object): the WorldStratDataset object itself

        Return:
        Dictionary: associates a number (index) to each label (class)
        """
        return self.__class_to_idx

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        "__getitem__": gets an image after receiving its index

        Args:
        self (WorldStratDataset Object): the WorldStratDataset object itself
        index (int): position of the element

        Return (tuple of torch tensors):
        img_transformed (torch tensor): transformed image
        (torch tensor): image class converted to tensor
        """
        folder_name = self.__metadata.iloc[index, 0]
        png_name = join(
            self.__root_path, folder_name, f"{folder_name}_rgb.png"
        )

        png_class = self.__metadata.iloc[index, 1]
        png_class_idx = self.class_to_idx[png_class]

        img = self.__img_loader(png_name)
        img_transformed = self.__img_transform(img)

        return img_transformed, th.tensor(png_class_idx)

    def __len__(self) -> int:
        """
        "__len__": returns the lenght (number of images) of the dataset

        Args:
        self (WorldStratDataset): WorldStratDataset object itself

        Return:
        int: number of images in the dataset
        """
        return len(self.__metadata)


class SkinCancerDataset(ImageFolder):
    """
    Creates the class of the SkinCancerDataset, extending the class ImageFolder
    """

    # https://github.com/Ipsedo/MARLClassification/issues/4
    # https://drive.google.com/drive/folders/17g6zFSbCNXTV3VaDKop73W7Cn-NJlTO7?usp=sharing
    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": constructor of the SkinCancerDataset class

        Args:
        self (SkinCancerDataset object): the SkinCancerDataset object itself
        res_path (str): SkinCancerDataset folder's path
        img_transform (callable): image transformation applied

        Return:
        None
        """
        skin_cancer_dataset_path = join(res_path, "downloaded", "skin_cancer")

        assert exists(skin_cancer_dataset_path) and isdir(
            skin_cancer_dataset_path
        ), f"{skin_cancer_dataset_path} does not exist or is not a directory"

        super().__init__(
            skin_cancer_dataset_path,
            transform=img_transform,
            target_transform=None,
            loader=my_pil_loader,
            is_valid_file=None,
        )


class KineticsDataset(Dataset):
    """
    Creates the KineticsDataset class extending from the Dataset class
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": KineticsDataset class constructor

        Args:
        self (KineticsDataset object): KineticsDataset object itself
        res_path (str): KineticsDataset folder's path
        img_transform (callable): image transformation applied

        Return:
        None
        """
        kinetics_dataset_path = join(
            res_path, "downloaded", "kinetics700_2020"
        )

        assert exists(kinetics_dataset_path) and isdir(kinetics_dataset_path)

        self.__videos_path = join(kinetics_dataset_path, "videos")

        train_df = pd.read_csv(join(kinetics_dataset_path, "train.csv"))
        video_ids = set(train_df["youtube_id"].tolist())

        self.__all_videos = [
            splitext(basename(f))[0]
            for f in glob.glob(join(self.__videos_path, "*.mp4"))
            if splitext(basename(f))[0] in video_ids
        ]

        tmp_all_video = set(self.__all_videos)
        self.__all_labels = {
            row["youtube_id"]: row["label"]
            for _, row in train_df.iterrows()
            if row["youtube_id"] in tmp_all_video
        }

        self.__class_to_idx = {
            label: i for i, label in enumerate(train_df["label"].unique())
        }

        self.__transform = img_transform

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        "__getitem__": gets an image after receiving an index

        Args:
        self (KineticsDataset object): KineticsDataset object itself
        index (int): image's index

        Return:
        tuple of torch tensors: the transformed image and the
        label of the image
        """
        video_path = self.__all_videos[index]
        video_label = self.__all_labels[video_path]

        video = torchvision.io.VideoReader(
            join(self.__videos_path, video_path + ".mp4"), "video"
        )

        video.set_current_stream("video")

        video_data = th.stack([frame["data"] for frame in video], dim=-1)

        return self.__transform(video_data), th.tensor(
            self.class_to_idx[video_label]
        )

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """
        "class_to_idx": creates a dictionary for each class associated with an index

        Args:
        self (KineticsDataset Object): the KineticsDataset object itself

        Return:
        Dictionary: associates a number (index) to each label (class)
        """
        return self.__class_to_idx

    def __len__(self) -> int:
        """
        "__len__": returns the lenght (number of images) of the dataset

        Args:
        self (KineticsDataset): KineticsDataset object itself

        Return:
        int: number of images in the dataset
        """
        return len(self.__all_videos)


class Cifar10Dataset(ImageFolder):
    """
    Class of the Cifar10 dataset, extending the ImageFolder object
    """

    def __init__(
        self, res_path: str, img_transform: Callable[[Any], th.Tensor]
    ) -> None:
        """
        "__init__": Cifar10Dataset class constructor

        Args:
        self (Cifar10Dataset Object): the Cifar10Dataset Object itself
        res_path (str): root path
        img_transform (callable): image transformation

        Return:
        None
        """

        cifar_root_path = join(res_path, "downloaded", "cifar_10", "all_png")

        assert exists(cifar_root_path) and isdir(
            cifar_root_path
        ), f"{cifar_root_path} does not exist or is not a directory"

        super().__init__(
            cifar_root_path,
            transform=img_transform,
            target_transform=None,
            loader=my_pil_loader,
            is_valid_file=None,
        )
