# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from typing import Tuple, cast

import torch as th

"""
The classes in this file are never called in the project
"""

class ImgTransform(metaclass=ABCMeta):
    """
    The super class of the image transformations
    """
    @abstractmethod
    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        """
        ""__call__"" is an abstract method that will initate an error 
        if the called method is not implemented

        Args:
        self (ImgTransform object): the ImgTransform object itself
        img_data (torch tensor): image

        Return:
        torch tensor: transformed image

        """
        raise NotImplementedError(
            self.__class__.__name__
            + ".__call__ method is not implemented, must be overridden !"
        )

    def __repr__(self) -> str:
        """
        "__repr__": defines the class name, when the object is called

        Args:
        self (ImgTransform object): the ImgTransform object itself

        Return:
        str: the class name
        """
        return self.__class__.__name__ + "()"


#########################
# Normal normalization
#########################
class UserNormalNorm(ImgTransform):
    """
    Class of the image normalization according to the Normal (Gaussian) distribution
    """
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        """
        "__init__": initiates the UserNormalNorm class

        Args:
        self (UserNormalNorm object): the UserNormalNorm object itself
        mean (Tuple of three Floats): mean value of each channel
        std (Tuple of three Floats): standard deviation value of each channel

        Return:
        None

        """
        super().__init__()

        self.__mean = th.tensor(mean)
        self.__std = th.tensor(std)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        """
        "__call__": when the function is called, returns the transformed
        image

        Args:
        self (UserNormalNorm object): the UserNormalNorm object itself
        x (torch tensor): the image itself

        Return:
        torch tensor: the transformed image
        
        """
        return (x - self.__mean) / self.__std

    def __repr__(self) -> str:
        """
        "__repr__": defines the class name, when the object is called

        Args:
        self (UserNormalNorm object): the UserNormalNorm object itself

        Return:
        str: a string with the mean and the standatd deviation
        """
        return (
            self.__class__.__name__
            + f"(mean = {str(self.__mean)}, std = {str(self.__std)})"
        )


class ChannelNormalNorm(ImgTransform):
    """
    Class defining the channel normalization with the Normal distribution
    applied channel-wise
    """
    def __call__(self, x: th.Tensor) -> th.Tensor:
        """
        "__call__": when the function is called, returns the transformed
        image

        Args:
        self (ChannelNormalNorm object): the ChannelNormalNorm object itself
        x (torch tensor): the image itself

        Return:
        torch tensor: the transformed image
        """
        mean = x.view(3, -1).mean(dim=-1).view(3, 1, 1)
        std = x.view(3, -1).std(dim=-1).view(3, 1, 1)

        return (x - mean) / std


class NormalNorm(ImgTransform):
    """
    Class of the image normalization according to the Normal (Gaussian) distribution
    """
    def __call__(self, x: th.Tensor) -> th.Tensor:
        """
        "__call__": when the function is called, returns the transformed
        image

        Args:
        self (NormalNorm object): the NormalNorm object itself
        x (torch tensor): the image itself

        Return:
        torch tensor: the transformed image
        """
        return (x - th.mean(x)) / th.std(x)


#########################
# Uniform normalization
#########################
class UserMinMaxNorm(ImgTransform):
    def __init__(
        self,
        min_value: Tuple[float, float, float],
        max_value: Tuple[float, float, float],
    ):
        self.__min = th.tensor(min_value)
        self.__max = th.tensor(max_value)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__min) / (self.__max - self.__min)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(min_value = {self.__min}, max_value = {self.__max})"
        )


class MinMaxNorm(ImgTransform):
    def __call__(self, x: th.Tensor) -> th.Tensor:
        x_max = x.max()
        x_min = x.min()
        return (x - x_min) / (x_max - x_min)


class ChannelMinMaxNorm(ImgTransform):
    def __call__(self, x: th.Tensor) -> th.Tensor:
        x_max = x.view(3, -1).max(dim=-1)[0].view(3, 1, 1)
        x_min = x.view(3, -1).min(dim=-1)[0].view(3, 1, 1)
        return cast(th.Tensor, (x - x_min) / (x_max - x_min))
