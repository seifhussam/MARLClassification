# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from os.path import join
from statistics import mean
from typing import Any, Generic, List, Mapping, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import torch as th


def format_metric(metric: th.Tensor, class_map: Mapping[Any, int]) -> str:
    """
        "format_metric" receives metrics from various classes and displays them accordingly

        Args:
        metric (torch Tensor): tensor with all the metrics for various classes
        class_map (typing Mapping): map that associates a class to an integer

        Return:
        str: message with metrics to each class
    """

    idx_to_class = {class_map[k]: k for k in class_map}

    return ", ".join(
        [
            f'"{idx_to_class[curr_cls]}" : {metric[curr_cls] * 100.:.1f}%'
            for curr_cls in range(metric.size()[0])
        ]
    )


# Generic types initiation

T = TypeVar("T")
I = TypeVar("I")


class Meter(Generic[I, T], ABC):

    """
    Used to initiate the class Meter, where I and T are generic data types that are later changed
    and ABC informs that this class functions can only be called by another subclass (e.g. ConfusionMeter)
    """

    def __init__(self, window_size: Optional[int]) -> None:

        """
        "__init__" is the Class constructor and initates the class

        Args:
        self (Meter object): the object Meter itself
        window_size (int): optional window size

        Returns:
        None
        """

        self.__window_size = window_size

        self.__results: List[T] = []

    @abstractmethod
    def _process_value(self, *args: I) -> T:
        """
        "_process_value" because it is an abstract method, it will be called later by a subclass

        Args:
            self (Meter object): the object Meter itself
            *args: multiple args of undeclared type
        
        Returns:
            Undeclared type
        """
        pass

    @property
    def _results(self) -> List[T]:
        """
        "_results": Read only property that returns the list of results

        Args: 
            self (Meter object): the object Meter itself

        Returns:
            List of objects of Undeclared type
        """
        return self.__results

    def add(self, *args: I) -> None:
        """
        "add": adds an element to the results list. If the size already surpassed the window_size,
        the first element of the list is excluded

        Args:
            self (Meter object): the object Meter itself
            *args: multiple args of an undeclared type

        Return:
            None
        """

        if (
            self.__window_size is not None
            and len(self.__results) >= self.__window_size
        ):
            self.__results.pop(0)

        self.__results.append(self._process_value(*args))

    def set_window_size(self, new_window_size: Union[int, None]) -> None:
        """
        "set_window_size": defines the window size of the Metric object

        Args:
            self (Meter object): the object Meter itself
            new_window_size (int OR None): window size

        Returns:
            None
        """
        if new_window_size is not None:
            assert (
                new_window_size > 0
            ), f"window size must be > 0 : {new_window_size}"

        self.__window_size = new_window_size


class ConfusionMeter(Meter[th.Tensor, Tuple[th.Tensor, th.Tensor]]):
    """
    Used to initiate the class ConfusionMeter, receiving as arguments one
    Meter object, declared with one tensor as one variable (I) and a tuple of
    tensors as the second variable (T)
    
    """

    def __init__(
        self,
        nb_class: int,
        window_size: Optional[int] = None,
    ):
        """
        "__init__" is the ConfusionMeter class constructor 

        Args:
            self (ConfusionMeter object): information about the object itself
            nb_class (int): number of classes
            window_size (int OR None): window size
        """
        super().__init__(window_size)
        self.__nb_class = nb_class

    def _process_value(self, *args: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        "_process_value" is a function that processes two torch tensors, 
        one with the probability of each class and one with the true label

        Args:
        self (ConfusionMeter object): information about the object itself
        *args (torch tensors): probability of each class and true label

        Returns:
        tuple: predicted label and the true label
        """
        y_proba = args[0]
        y_true = args[1]
        return y_proba.argmax(dim=1), y_true

    def conf_mat(self) -> th.Tensor:
        """
        "conf_mat" calculates the confusion matrix

        Args:
        self (ConfusionMeter object): information about the object itself

        Returns:
        torch Tensor: confusion matrix
        """
        y_pred = th.cat([y_p for y_p, _ in self._results], dim=0)
        y_true = th.cat([y_t for _, y_t in self._results], dim=0)

        conf_matrix_indices = th.multiply(y_true, self.__nb_class) + y_pred
        conf_matrix = th.bincount(
            conf_matrix_indices, minlength=self.__nb_class**2
        ).reshape(self.__nb_class, self.__nb_class)

        return conf_matrix

    def precision(self) -> th.Tensor:
        """
        "precision" calculates the precision

        Args:
        self (ConfusionMeter object): information about the object itself

        Returns:
        torch Tensor: precision for each class
        """
        conf_mat = self.conf_mat()

        precs_sum = conf_mat.sum(dim=0)
        diag = th.diagonal(conf_mat, 0)

        precs = th.zeros(self.__nb_class, device=conf_mat.device)

        mask = precs_sum != 0

        precs[mask] = diag[mask] / precs_sum[mask]

        return precs

    def recall(self) -> th.Tensor:
        """
        "recall" calculates the recall

        Args:
        self (ConfusionMeter object): information about the object itself

        Returns:
        torch Tensor: recall for each class
        """
        conf_mat = self.conf_mat()

        recs_sum = conf_mat.sum(dim=1)
        diag = th.diagonal(conf_mat, 0)

        recs = th.zeros(self.__nb_class, device=conf_mat.device)

        mask = recs_sum != 0

        recs[mask] = diag[mask] / recs_sum[mask]

        return recs

    def save_conf_matrix(
        self, epoch: int, output_dir: str, stage: str
    ) -> None:
        
        """
        "save_conf_matrix" saves an image of the confusion matrix

        Args:
        self (ConfusionMeter object): information about the object itself
        epoch (int): epoch number at the time of the image
        output_dir (str): output directory of the images
        stage (str): defines whether it is the confusion matrix in training or evaluation

        Returns:
        None
        """
                
        fig = plt.figure()
        ax = fig.add_subplot(111)

        conf_mat = self.conf_mat()
        conf_mat_normalized = conf_mat / th.sum(conf_mat, dim=1, keepdim=True)
        cax = ax.matshow(conf_mat_normalized.tolist(), cmap="plasma")
        fig.colorbar(cax)

        ax.set_title(f"confusion matrix epoch {epoch} - {stage}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicated Label")

        fig.savefig(
            join(output_dir, f"confusion_matrix_epoch_{epoch}_{stage}.png")
        )

        plt.close()


class LossMeter(Meter[float, float]):
    """
    Used to initiate the class LossMeter, receiving as arguments one
    Meter object, declared with two floats
    
    """

    def _process_value(self, *args: float) -> float:
        """
        "_process_value" is a function that processes multiple floats
        and returns the first last one added

        Args:
        self (ConfusionMeter object): information about the object itself
        *args (float): loss as a float

        Returns:
        float: last inserted loss
        """
        return args[0]

    def loss(self) -> float:
        """
        "loss" is a function that processes returns the loss as a float
        resulting from the mean of the losses inserted

        Args:
        self (ConfusionMeter object): information about the object itself

        Returns:
        float: mean of the loss
        """
        return mean(self._results)