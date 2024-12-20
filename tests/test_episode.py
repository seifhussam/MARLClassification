# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th

from marl_classification.core import MultiAgent, detailed_episode, episode


def test_episode(
    batch_size: int,
    marl_m: MultiAgent,
    step: int,
    nb_class: int,
    nb_agent: int,
    height_width: Tuple[int, int],
) -> None:
    """
    "test_episode": used to test the results of a mockup episode
    
    Args:
    batch_size (int): size of the batch
    marl_m (MultiAgent object): object with the multiple agents in the
    simulation
    step (int): number of steps
    nb_class (int): number of classes
    nb_agent (int): number of agents
    height_width (Tuple[int, int]): height and width of the image

    Return:
    None
    """

    x = th.randn(batch_size, 1, *height_width)

    pred, log_proba = episode(marl_m, x, step)

    assert 3 == len(pred.size())
    assert nb_agent == pred.size()[0]
    assert batch_size == pred.size()[1]
    assert nb_class == pred.size()[2]

    assert 2 == len(log_proba.size())
    assert nb_agent == log_proba.size()[0]
    assert batch_size == log_proba.size()[1]


def test_detailed_episode(
    batch_size: int,
    marl_m: MultiAgent,
    step: int,
    nb_class: int,
    nb_agent: int,
    dim: int,
    height_width: Tuple[int, int],
) -> None:
    """
    "test_detailed_episode": used to test the results of a mockup episode
    displaying detailed information
    
    Args:
    batch_size (int): size of the batch
    marl_m (MultiAgent object): object with the multiple agents in the simulation
    step (int): number of steps
    nb_class (int): number of classes
    nb_agent (int): number of agents
    dim (int): dimensions of the dataset (2 if it is 2D and 3 if it is 3D)
    height_width (Tuple[int, int]): height and width of the image

    Return:
    None
    """

    x = th.randn(batch_size, 1, *height_width)

    pred, log_proba, values, pos = detailed_episode(
        marl_m,
        x,
        step,
        "cpu",
        nb_class,
    )

    assert 4 == len(pred.size())
    assert step == pred.size()[0]
    assert nb_agent == pred.size()[1]
    assert batch_size == pred.size()[2]
    assert nb_class == pred.size()[3]

    assert 3 == len(log_proba.size())
    assert step == log_proba.size()[0]
    assert nb_agent == log_proba.size()[1]
    assert batch_size == log_proba.size()[2]

    assert 3 == len(values.size())
    assert step == values.size()[0]
    assert nb_agent == values.size()[1]
    assert batch_size == values.size()[2]

    assert 4 == len(pos.size())
    assert step == pos.size()[0]
    assert nb_agent == pos.size()[1]
    assert batch_size == pos.size()[2]
    assert dim == pos.size()[3]
