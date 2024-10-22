# -*- coding: utf-8 -*-
from typing import cast

import torch as th
from torch import nn
from torchvision.ops import Permute


class Policy(nn.Module):
    """
    Creates the Policy class, extending from the nn.Module
    """
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """

    def __init__(self, nb_action: int, n: int, hidden_size: int) -> None:
        """
        "__init__": Policy class constructor

        Args:
        self (Policy class): Policy class itself
        nb_action (int): number of actions
        n (int): input shape
        hidden_size (int): size of the hidden layers

        Return:
        None
        """
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1),
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the Policy network

        Args:
        self (Policy class): Policy class itself
        h_caret_t_next (torch tensor): network input

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_lin(h_caret_t_next))


class Critic(nn.Module):
    """
    Create Critic class extending nn.Module
    """
    def __init__(self, n: int, hidden_size: int):
        """
        "__init__": Critic class constructor

        Args:
        self (Critic class): Critic class itself
        nb_action (int): number of actions
        n (int): input shape
        hidden_size (int): size of the hidden layers

        Return:
        None
        """
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, 1),
            nn.Flatten(-2, -1),
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        """
        "forward": forward step of the Critic network

        Args:
        self (Critic class): Critic class itself
        h_caret_t_next (torch tensor): network input

        Return:
        torch tensor: result of the forward step
        """
        return cast(th.Tensor, self.__seq_lin(h_caret_t_next))
