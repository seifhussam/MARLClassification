# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import cast

import torch as th
from torch import nn
from torchvision.ops import Permute


class CNNFtExtract(nn.Module, ABC):
    """
    Class of the CNNs used for feature extraction, extending torch nn.Module 
    """
    @property
    @abstractmethod
    def out_size(self) -> int:
        raise NotImplementedError()


############################
# Features extraction stuff
############################

# MNIST Stuff


class MNISTCnn(CNNFtExtract):
    """
    Class of the MNIST feature extracting, extending CNNFtExtract
    """
    """
    b_θ5 : R^f*f -> R^n
    """

    def __init__(self, f: int) -> None:
        """
        "__init__": MNISTCnn constructor

        Args:
        self (MNISTCnn object): MNISTCnn object itself
        f (int): window size
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 4) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (MNISTCnn object): MNISTCnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step

        """
        o_t = o_t[:, 0, None, :, :]  # grey scale
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (MNISTCnn object): MNISTCnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size


# RESISC-45 Stuff


class RESISC45Cnn(CNNFtExtract):
    """
    Class of the RESISC45 feature extracting, extending CNNFtExtract
    """
    def __init__(self, f: int) -> None:
        """
        "__init__": RESISC45Cnn constructor

        Args:
        self (RESISC45Cnn object): RESISC45Cnn object itself
        f (int): window size
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Flatten(1, -1),
        )

        self.__out_size = 64 * (f // 8) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (RESISC45Cnn object): RESISC45Cnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (RESISC45Cnn object): RESISC45Cnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size


class AIDCnn(CNNFtExtract):
    """
    Class of the AID feature extracting, extending CNNFtExtract
    """
    def __init__(self, f: int) -> None:
        """
        "__init__": AIDCnn constructor

        Args:
        self (AIDCnn object): AIDCnn object itself
        f (int): window size
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Flatten(1, -1),
        )

        self.__out_size = 128 * (f // 16) ** 2

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (AIDCnn object): AIDCnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (AIDCnn object): AIDCnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_conv(o_t))


class WorldStratCnn(CNNFtExtract):
    """
    Class of the WorldStrat feature extracting, extending CNNFtExtract
    """
    def __init__(self, f: int) -> None:
        """
        "__init__": WorldStratCnn constructor

        Args:
        self (WorldStratCnn object): WorldStratCnn object itself
        f (int): window size
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.Flatten(1, -1),
        )

        self.__out_size = 256 * (f // 32) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (WorldStratCnn object): WorldStratCnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_conv(o_t))

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (WorldStratCnn object): WorldStratCnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size


# Knee MRI stuff


class KneeMRICnn(CNNFtExtract):
    """
    Class of the KneeMRI feature extracting, extending CNNFtExtract
    """
    def __init__(self, f: int = 16):
        """
        "__init__": KneeMRICnn constructor

        Args:
        self (KneeMRICnn object): KneeMRICnn object itself
        f (int): window size, default 16
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(8),
            nn.Conv3d(8, 16, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 8) ** 3

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (KneeMRICnn object): KneeMRICnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        out = cast(th.Tensor, self.__seq_conv(o_t))
        return out

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (KneeMRICnn object): KneeMRICnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size


class SkinCancerCnn(CNNFtExtract):
    """
    Class of the SkinCancer feature extracting, extending CNNFtExtract
    """
    # https://github.com/Ipsedo/MARLClassification/issues/4
    # https://drive.google.com/drive/folders/17g6zFSbCNXTV3VaDKop73W7Cn-NJlTO7?usp=sharing
    def __init__(self, f: int) -> None:
        """
        "__init__": SkinCancerCnn constructor

        Args:
        self (SkinCancerCnn object): SkinCancerCnn object itself
        f (int): window size
        
        Return: None
        """
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Flatten(1, -1),
        )

        self.__out_size = 64 * (f // 8) ** 2

    @property
    def out_size(self) -> int:
        """
        "out_size": size of the output

        Args:
        self (KneeMRICnn object): KneeMRICnn object itself

        Return:
        int: size of the output
        """
        return self.__out_size

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (KneeMRICnn object): KneeMRICnn object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        out: th.Tensor = self.__seq_conv(o_t)
        return out


############################
# State to features stuff
############################
class StateToFeatures(nn.Module):
    """
    Creates the StateToFeatures class, which extends torch nn.Module
    """
    """
    λ_θ7 : R^d -> R^n
    """

    def __init__(self, d: int, n_d: int) -> None:
        """
        "__init__": StateToFeatures class constructor

        Args:
        d (int): one dimension of the input
        n_d (int): another dimension of the input

        Return:
        None
        """
        super().__init__()

        self.__d = d
        self.__n_d = n_d

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__d, self.__n_d),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.__n_d),
            Permute([2, 0, 1]),
        )

    def forward(self, p_t: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the CNN

        Args:
        self (StateToFeatures object): StateToFeatures object itself
        o_t (torch tensor): image input of the CNN

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_lin(p_t))
