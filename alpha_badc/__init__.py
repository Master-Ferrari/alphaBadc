"""AlphaBADC — состязательное обучение AlphaZero для игры BADC (TensorFlow)."""

from .config import AZConfig, GameConfig
from .game import BADCGame

__all__ = ["AZConfig", "GameConfig", "BADCGame"]
