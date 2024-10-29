# -*- coding: utf-8 -*-
import json
from os.path import exists, isfile
from typing import Callable, Dict, List, Set, Tuple, Union, cast

import torch as th
from torch import nn

from .ft_extractor import (
    AIDCnn,
    CNNFtExtract,
    KneeMRICnn,
    MNISTCnn,
    RESISC45Cnn,
    SkinCancerCnn,
    StateToFeatures,
    WorldStratCnn,
)
from .message import MessageSender, MessageReceiver
from .policy import Critic, Policy
from .prediction import Prediction
from .recurrent import LSTMCellWrapper


#####################
# Base class test
#####################
class ModelsWrapper(nn.Module):
    """
    Most important class in the project, ModelsWrapper, which extends the nn.Module

    It starts with a dictionary that stores the corresponding name of each equation
    in the paper to the code module.

    Then creates a dictionary for the possible datasets to use this implementation
    """
    # Modules
    map_obs: str = "b_theta_5"
    map_pos: str = "lambda_theta_7"

    evaluate_msg: str = "m_theta_4"

    receiver_msg: str = "d_theta_6"

    belief_unit: str = "belief_unit"
    action_unit: str = "action_unit"

    policy: str = "pi_theta_3"
    predict: str = "q_theta_8"

    critic: str = "critic"

    module_list: Set[str] = {
        map_obs,
        map_pos,
        evaluate_msg,
        receiver_msg,
        belief_unit,
        action_unit,
        policy,
        critic,
        predict,
    }

    # Features extractors - CNN
    mnist: str = "mnist"
    resisc: str = "resisc45"
    knee_mri: str = "kneemri"
    aid: str = "aid"
    world_strat: str = "worldstrat"
    skin_cancer: str = "skin_cancer"

    ft_extractors: Dict[str, Callable[[int], CNNFtExtract]] = {
        mnist: MNISTCnn,
        resisc: RESISC45Cnn,
        knee_mri: KneeMRICnn,
        aid: AIDCnn,
        world_strat: WorldStratCnn,
        skin_cancer: SkinCancerCnn,
    }

    def __init__(
        self,
        ft_extr_str: str,
        f: int,
        n_b: int,
        n_a: int,
        n_m: int,
        n_d: int,
        d: int,
        actions: List[List[int]],
        nb_class: int,
        hidden_size_belief: int,
        hidden_size_action: int,
    ) -> None:
        """
        "__init__": ModelsWrapper class constructor that uses as input all the 
        necessary information in the training or testing process

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself
        ft_extr_str (str): string describing the feature extracting module used
        f (int): window size
        n_b (int): hidden size for belief in LSTM
        n_a (int): hidden size for action LTSM
        n_m (int): message size for Neural Networks
        n_d (int): state hidden size
        d (int): state dimension
        actions (List[List[int]]): list of possible actions
        nb_class (int): number of possible classes
        hidden_size_belief (int): size of the hidden belief
        hidden_size_action (int): size of the hidden action

        Return: None
        """
        super().__init__()

        map_obs_module = self.ft_extractors[ft_extr_str](f)

        self.__networks_dict = nn.ModuleDict(
            {
                self.map_obs: map_obs_module, # Agent partial observation
                self.map_pos: StateToFeatures(d, n_d), # Processes the position of the agent to features
                self.evaluate_msg: MessageSender(n_b, n_m, hidden_size_belief), # one component of Communication module
                self.receiver_msg: MessageReceiver(n_m, n_b), # one component of Communication module
                self.belief_unit: LSTMCellWrapper(
                    map_obs_module.out_size + n_d + n_b, n_b
                ), # belief Module
                # Input: result of the partial observation, agent position, and message received (In the article, the 
                # aggregate of this three metrics correspond to the u letter)
                # hidden (h) and cell (c) state in the equation belong yo the LSTMCellWrapper
                # Equation 1 
                self.action_unit: LSTMCellWrapper(
                    map_obs_module.out_size + n_d + n_b, n_a
                ), # Decision Module
                # Input: result of the partial observation, agent position, and message received (In the article, the 
                # aggregate of this three metrics correspond to the u letter)
                # hidden (h) and cell (c) state in the equation belong yo the LSTMCellWrapper
                # Equation 4
                self.policy: Policy(len(actions), n_a, hidden_size_action), # Policy Module
                self.critic: Critic(n_a, hidden_size_action),
                self.predict: Prediction(n_b, nb_class, hidden_size_belief), # Prediction Module
            }
        )

        self.__ft_extr_str = ft_extr_str

        self.__f = f
        self.__n = n_b
        self.__n_a = n_a
        self.__n_l_b = hidden_size_belief
        self.__n_l_a = hidden_size_action
        self.__n_m = n_m
        self.__n_d = n_d

        self.__d = d
        self.__actions = actions
        self.__nb_class = nb_class

        def __init_weights(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(__init_weights)

    def forward(
        self, op: str, *args: th.Tensor
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        "forward": calculates the forward step of the ModelsWrapper

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself
        op (str): name of the network to use
        *args (torch tensor): multiple unspecified arguments that are the
        networks input

        Return:
        Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]: results of the forward
        pass in each network
        """
        return cast(
            Union[th.Tensor, Tuple[th.Tensor, th.Tensor]],
            self.__networks_dict[op](*args),
        )

    @property
    def nb_class(self) -> int:
        """
        "nb_class": returns the number of classes expected in the ModelsWrapper

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself

        Return:
        int: number of classes
        """
        return self.__nb_class

    @property
    def f(self) -> int:
        """
        "f": returns the window size in the ModelsWrapper

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself

        Return:
        int: window size
        """
        return self.__f

    def get_params(self, ops: List[str]) -> List[th.Tensor]:
        """
        "get_params": returns the number of parameters in the desired networks

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself
        ops (List[str]): desired networks

        Return:
        List[torch tensor]: list with the number of parameters in each network
        """
        return [p for op in ops for p in self.__networks_dict[op].parameters()]

    def json_args(self, out_json_path: str) -> None:
        """
        "json_args": saves the arguments used in the model to a JSON file

        Args:
        self (ModelsWrapper object): ModelsWrapper object itself
        out_json_path (str): output path

        Return:
        None
        """
        with open(out_json_path, "w", encoding="utf-8") as json_f:
            args_d = {
                "ft_extr_str": self.__ft_extr_str,
                "window_size": self.__f,
                "hidden_size_belief": self.__n,
                "hidden_size_action": self.__n_a,
                "hidden_size_msg": self.__n_m,
                "hidden_size_state": self.__n_d,
                "state_dim": self.__d,
                "actions": self.__actions,
                "class_number": self.__nb_class,
                "hidden_size_linear_belief": self.__n_l_b,
                "hidden_size_linear_action": self.__n_l_a,
            }

            json.dump(args_d, json_f)

    @classmethod
    def from_json(cls, json_path: str) -> "ModelsWrapper":
        """
        "from_json": reads a model from a JSON file

        Args:
        cls: ModelsWrapper class itself
        json_path (str): JSON file path

        Return:
        ModelsWrapper object
        """
        assert exists(json_path) and isfile(
            json_path
        ), f'"{json_path}" does not exist or is not a file'

        with open(json_path, "r", encoding="utf-8") as json_f:
            args_d = json.load(json_f)

            return cls(
                args_d["ft_extr_str"],
                args_d["window_size"],
                args_d["hidden_size_belief"],
                args_d["hidden_size_action"],
                args_d["hidden_size_msg"],
                args_d["hidden_size_state"],
                args_d["state_dim"],
                args_d["actions"],
                args_d["class_number"],
                args_d["hidden_size_linear_belief"],
                args_d["hidden_size_linear_action"],
            )
