from torch import optim
import typing as ty
from abc import abstractmethod

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR, _LRScheduler

from ablator.config.main import ConfigBase, Derived, configclass
from ablator.config.types import Literal
from unittest.mock import MagicMock
import pytest
from ablator.modules.storage.remote import OneCycleConfig


class TestOneCycleConfig:
    def test_init_scheduler(self):
        model = MagicMock()
        optimizer = optim.Adam(model.parameters())
        expected = config.init_scheduler(model, optimizer)
        assert isinstance(expected, OneCycleLR)
        assert expected.max_lr == 1
        assert expected.total_steps == 1
