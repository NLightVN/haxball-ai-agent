"""
config.py - Cấu hình cho huấn luyện Elo-based

Chứa các hyperparameters và cấu hình cho training curriculum.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CurriculumConfig:
    """Cấu hình curriculum."""
    initial_pool_size: int = 10        # Số agents khởi đầu
    add_per_cycle: int = 10            # Số agents thêm mỗi cycle
    max_pool_size: int = 100           # Kích thước pool tối đa
    initial_num_cycles: int = 10       # Số cycles để chạy lần đầu


@dataclass
class PPOConfig:
    """Cấu hình huấn luyện PPO."""
    learning_rate: float = 3e-4        # Learning rate
    gamma: float = 0.99                # Discount factor
    gae_lambda: float = 0.95           # GAE lambda
    epsilon: float = 0.2               # PPO clip range
    value_coef: float = 0.5            # Value loss coefficient
    entropy_coef: float = 0.01         # Entropy bonus
    max_grad_norm: float = 0.5         # Gradient clipping
    num_epochs: int = 4                # Số epochs per update
    batch_size: int = 32               # Batch size
    device: str = "cuda"               # Device (cuda/cpu)


@dataclass
class EloConfig:
    """Cấu hình Elo rating system."""
    initial_elo: float = 1600.0        # Elo khởi đầu
    k_base: float = 64.0               # K-factor cơ bản


@dataclass
class MatchConfig:
    """Cấu hình mỗi trận đấu."""
    num_episodes: int = 5              # Số episodes per match
    time_limit_min: int = 3            # Thời gian giới hạn (phút)
    render: bool = False               # Có render hay không
    seed: Optional[int] = None         # Random seed


@dataclass
class TrainingConfig:
    """Tổng hợp cấu hình huấn luyện."""
    curriculum: CurriculumConfig = None
    ppo: PPOConfig = None
    elo: EloConfig = None
    match: MatchConfig = None
    
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    model_save_interval: int = 1       # Save model mỗi N cycles
    
    def __post_init__(self):
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()
        if self.ppo is None:
            self.ppo = PPOConfig()
        if self.elo is None:
            self.elo = EloConfig()
        if self.match is None:
            self.match = MatchConfig()


# Cấu hình mặc định cho các mode khác nhau

QUICK_DEBUG_CONFIG = TrainingConfig(
    curriculum=CurriculumConfig(
        initial_pool_size=4,
        add_per_cycle=2,
        max_pool_size=10,
        initial_num_cycles=2,
    ),
    match=MatchConfig(
        num_episodes=1,
        render=False,
    ),
)

STANDARD_CONFIG = TrainingConfig(
    curriculum=CurriculumConfig(
        initial_pool_size=10,
        add_per_cycle=10,
        max_pool_size=100,
        initial_num_cycles=10,
    ),
)

LARGE_SCALE_CONFIG = TrainingConfig(
    curriculum=CurriculumConfig(
        initial_pool_size=20,
        add_per_cycle=20,
        max_pool_size=200,
        initial_num_cycles=20,
    ),
)


def get_config(mode: str = "standard") -> TrainingConfig:
    """
    Lấy training config theo mode.
    
    Modes:
        debug: Quick debugging
        standard: Cấu hình tiêu chuẩn
        large: Huấn luyện quy mô lớn
    """
    if mode.lower() == "debug":
        return QUICK_DEBUG_CONFIG
    elif mode.lower() == "standard":
        return STANDARD_CONFIG
    elif mode.lower() == "large":
        return LARGE_SCALE_CONFIG
    else:
        raise ValueError(f"Unknown mode: {mode}")
