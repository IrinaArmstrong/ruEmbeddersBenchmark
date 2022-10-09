import torch
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, Dict

from ru_encoders_benchmark.logging_handler import get_logger


@typechecked
class Task(ABC):
    """
    An abstract class for all tasks.
    """

    def __init__(self, task_name: str):
        super().__init__()
        self.task_name = task_name
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

    @abstractmethod
    def train(self, embedder: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def eval(self,
             embedder: torch.nn.Module,
             model_name: str) -> Tuple[Optional[float], Dict[str, Any]]:
        raise NotImplementedError
