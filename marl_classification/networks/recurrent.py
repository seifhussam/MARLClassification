# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th
from torch import nn


class LSTMCellWrapper(nn.Module):
    """
    Create the LTSMCellWrapper extending nn.Module
    """
    # f_θ1 : R^2n * R^3n -> R^2n
    #
    # f_θ2 : ?
    # Supposition : R^2n * R^3n -> R^2n
    # R^2n : pas sûr

    def __init__(self, input_size: int, n: int) -> None:
        """
        "__init__": LSTMCellWrapper constructor

        Args:
        self (LSTMCellWrapper object): LSTMCellWrapper object itself
        input_size (int): size of the input
        n (int): hidden size

        Return:
        None
        """
        super().__init__()

        self.__lstm = nn.LSTMCell(input_size, n)

    def forward(
        self, h: th.Tensor, c: th.Tensor, u: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        "forward": forward step of the LSTMCell

        Args:
        self (LSTMCellWrapper object): LSTMCellWrapper object itself
        h (torch tensor): input feature dimension
        c (torch tensor): input feature dimension
        u (torch tensor): input size
        
        Return:
        torch tensor: hidden state and cell state
        """
        nb_ag, batch_size, _ = h.size()

        h, c, u = (
            h.flatten(0, 1),
            c.flatten(0, 1),
            u.flatten(0, 1),
        )

        h_next, c_next = self.__lstm(u, (h, c))

        return (
            h_next.view(nb_ag, batch_size, -1),
            c_next.view(nb_ag, batch_size, -1),
        )
