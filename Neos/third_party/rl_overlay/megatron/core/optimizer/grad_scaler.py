# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron grad scaler."""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class MegatronGradScaler(ABC):
    def __init__(self, initial_scale: float):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0
        self._scale = torch.tensor([initial_scale], dtype=torch.float, device='cuda')

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()

    @abstractmethod
    def update(self, found_inf: bool):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        pass


class ConstantGradScaler(MegatronGradScaler):
    """
    Constant grad scaler (loss scale is never adjusted regardless of NaNs seen in gradients).
    """

    def update(self, found_inf: bool):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


class DynamicGradScaler(MegatronGradScaler):
    """
    Grad scaler with dynamic scale that gets adjusted during training.

    Reduces loss scale by `backoff_factor` if `hysteresis` number of NaNs are seen in a row. Increases
    loss scale by `growth_factor` if NaNs are not seen for `growth_interval` iterations.
    """

    def __init__(
        self,
        initial_scale: float,
        min_scale: float,
        growth_factor: float,
        backoff_factor: float,
        growth_interval: int,
        hysteresis: int,
    ):
        """
        Grad scaler with dynamic scale that gets adjusted during training.

        Args:
            initial_scale (float): Initial loss scale value.
            min_scale (float): Minimum loss scale value.
            growth_factor (float): Factor to grow loss scale by if NaNs are not seen in `growth_interval`
                training iterations. Must be greater than 1.
            backoff_factor (float): Factor to decrease loss scale by if NaNs are seen in `hysteresis`
                consecutive training iterations. Must be between 0 and 1.
            growth_interval (int): Number of training iterations of no NaNs before loss scale is increased.
            hysteresis (int): Number of training iterations of consecutive NaNs before loss scale is decreased.
        """
        super(DynamicGradScaler, self).__init__(initial_scale)

        
        assert min_scale > 0.0
        assert min_scale <= initial_scale
        self.min_scale = torch.tensor([min_scale], dtype=torch.float, device='cuda')
        
        assert growth_factor > 1.0
        self.growth_factor = torch.tensor([growth_factor], dtype=torch.float, device='cuda')
        assert backoff_factor < 1.0
        assert backoff_factor > 0.0
        self.backoff_factor = torch.tensor([backoff_factor], dtype=torch.float, device='cuda')
        
        
        assert growth_interval > 0
        self.growth_interval = growth_interval
        
        
        assert hysteresis > 0
        self.hysteresis = hysteresis

        
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis

    def update(self, found_inf: bool):
        """
        Updates internal state in grad scaler based on whether NaNs are seen in grads or not.
        """

        
        
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
        else:
            
            self._growth_tracker += 1
            
            if self._growth_tracker == self.growth_interval:
                
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                
                self._scale = self._scale * self.growth_factor

    def state_dict(self):
        state_dict = {}
        state_dict['scale'] = self._scale
        state_dict['growth_tracker'] = self._growth_tracker
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
