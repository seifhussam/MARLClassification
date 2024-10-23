# -*- coding: utf-8 -*-
import shutil
from os import mkdir
from os.path import abspath, exists, isdir, join
from typing import Tuple

import pytest
from pytest import Session

from marl_classification.core import MultiAgent, obs_generic, trans_generic
from marl_classification.networks import ModelsWrapper

__TMP_PATH = abspath(join(__file__, "..", "tmp"))


@pytest.fixture(scope="session", name="batch_size")
def get_batch_size() -> int:
    """
    "get_batch_size": declares the batch size

    Args:
    None

    Return:
    int (19): batch size
    """
    return 19


@pytest.fixture(scope="session", name="nb_agent")
def get_nb_agent() -> int:
    """
    "get_nb_agent": declares the number of agents

    Args:
    None

    Return:
    int (5): number of agents
    """
    return 5


@pytest.fixture(scope="session", name="nb_class")
def get_nb_class() -> int:
    """
    "get_nb_class": declares the number of classes

    Args:
    None

    Return:
    int (10): number of classes
    """
    return 10


@pytest.fixture(scope="session", name="step")
def get_step() -> int:
    """
    "get_step": declares the number of steps

    Args:
    None

    Return:
    int (7): number of steps
    """
    return 7


@pytest.fixture(scope="session", name="dim")
def get_dim() -> int:
    """
    "get_dim": declares the number of dimensions
    (2 if it is 2D and 3 if it is 3D)

    Args:
    None

    Return:
    int (2): number of dimensions
    """
    return 2


@pytest.fixture(scope="session", name="ft_extractor")
def get_ft_extractor() -> str:
    """
    "get_ft_extractor": declares the feature extractor

    Args:
    None

    Return:
    str ("mnist"): feature extractor used
    """
    return "mnist"


@pytest.fixture(scope="session", name="height_width")
def get_height_width() -> Tuple[int, int]:
    """
    "get_height_width": declares the height and width
    of the image

    Args:
    None

    Return:
    Tuple[int, int] (28, 28): height and width of the
    image
    """
    return 28, 28


@pytest.fixture(scope="session", name="marl_m")
def get_marl_m(
    dim: int, nb_class: int, nb_agent: int, ft_extractor: str
) -> MultiAgent:
    """
    "get_marl_m": function to test the implementation of the
    multi agent object with the previously inserted data

    Args:
    dim (int): dimensions of the dataset (2 if it is 2D and 3 if it is 3D)
    nb_class (int): number of classes
    nb_agent (int): number of agents
    ft_extractor: name of the feature extractor

    Return:
    MultiAgent object: complete MultiAgent object

    """
    action = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    n_b = 23
    n_a = 22
    n_m = 21

    f = 12

    model_wrapper = ModelsWrapper(
        ft_extractor,
        f,
        n_b,
        n_a,
        n_m,
        20,
        dim,
        action,
        nb_class,
        24,
        25,
    )

    return MultiAgent(
        nb_agent,
        model_wrapper,
        n_b,
        n_a,
        f,
        n_m,
        action,
        obs_generic,
        trans_generic,
    )


@pytest.fixture(scope="module", name="tmp_path")
def get_tmp_path() -> str:
    """
    "get_tmp_path": gets the defined temporary path

    Args: 
    None

    Return:
    __TMP_PATH (str): temporary path
    """
    return __TMP_PATH


# pylint: disable=(unused-argument)
def pytest_sessionstart(session: Session) -> None:
    """
    "pytest_sessionstart": initiates the testing session

    Args:
    session (Session): session

    Return:
    None
    """
    if not exists(__TMP_PATH):
        mkdir(__TMP_PATH)
    elif not isdir(__TMP_PATH):
        pytest.fail(f'"{__TMP_PATH}" is not a directory')


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """
    "pytest_sessionfinish": ends the testing session and removes 
    the contents in the temporary path and the folder itself

    Args:
    session (Session): session
    exitstatus (int): session exit status

    Return:
    None
    """
    shutil.rmtree(__TMP_PATH)


# pylint: enable=(unused-argument)
