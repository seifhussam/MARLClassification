# -*- coding: utf-8 -*-
from typing import cast

import torch as th
from torch import nn
from torchvision.ops import Permute


class MessageSender(nn.Module):
    """
    Creates the class responsible for sending messages between agents, extending nn.Module
    """
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int, hidden_size: int) -> None:
        """
        "__init__": MessageSender constructor

        Args:
        self (MessageSender object): MessageSender object itself
        n (int): input dimension
        n_m (int): message dimension
        hidden_size (int): hidden layer size

        Return:
        None
        """
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = hidden_size

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n_e),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.__n_e),
            Permute([2, 0, 1]),
            nn.Linear(self.__n_e, self.__n_m),
        )

    def forward(self, h_t: th.Tensor) -> th.Tensor:
        """
        "forward": returns the forward step of the network
        
        Args:
        self (MessageSender object): MessageSender object itself
        h_t (torch tensor): input tensor

        Return:
        torch tensor: forward step of the network
        """
        return cast(th.Tensor, self.__seq_lin(h_t))


class MessageReceiver(nn.Module):
    """
    Creates the MessageReceiver class, extending the nn.Module
    """
    """
    d_θ6 : R^n_m -> R^n
    """

    def __init__(self, n_m: int, n: int) -> None:
        """
        "__init__": MessageReceiver constructor

        Args:
        self (MessageReceiver object): MessageReceiver object itself
        n (int): input dimension
        n_m (int): message dimension

        Return:
        None
        """
        super().__init__()
        self.__n = n
        self.__n_m = n_m

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n_m, self.__n),
            nn.GELU(),
        )

    def forward(self, m_t: th.Tensor) -> th.Tensor:
        """
        "forward": returns the forward step of the network
        
        Args:
        self (MessageReceiver object): MessageReceiver object itself
        h_t (torch tensor): input tensor

        Return:
        torch tensor: forward step of the network
        """
        return cast(th.Tensor, self.__seq_lin(m_t))
