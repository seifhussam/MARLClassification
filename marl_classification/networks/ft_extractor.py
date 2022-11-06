from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn


class CNNFtExtract(nn.Module, ABC):

    @property
    @abstractmethod
    def out_size(self) -> int:
        return -1


############################
# Features extraction stuff
############################

# MNIST Stuff

class MNISTCnn(CNNFtExtract):
    """
    b_θ5 : R^f*f -> R^n
    """

    def __init__(self, f: int) -> None:
        super().__init__()

        self.__f = f

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(1, -1)
        )

        self.__out_size = 8 * f ** 2

    def forward(self, o_t):
        o_t = o_t[:, 0, None, :, :]  # grey scale
        return self.__seq_conv(o_t)

    @property
    def out_size(self) -> int:
        return self.__out_size


# RESISC-45 Stuff

class RESISC45Cnn(CNNFtExtract):

    def __init__(self, f: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1, -1)
        )

        self.__out_size = 32 * (f // 4) ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        return self.__seq_conv(o_t)

    @property
    def out_size(self) -> int:
        return self.__out_size


# Knee MRI stuff

class KneeMRICnn(CNNFtExtract):
    def __init__(self, f: int = 16):
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(4, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Flatten(1, -1)
        )

        self.__out_size = 16 * (f // 8) ** 3

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = self.__seq_conv(o_t)
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


############################
# State to features stuff
############################
class StateToFeatures(nn.Module):
    """
    λ_θ7 : R^d -> R^n
    """

    def __init__(self, d: int, n_d: int) -> None:
        super().__init__()

        self.__d = d
        self.__n_d = n_d

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__d, self.__n_d),
            nn.ReLU()
        )

    def forward(self, p_t):
        return self.seq_lin(p_t)