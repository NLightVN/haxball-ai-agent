"""
HaxBall AI Training - Multi-Agent PPO
=====================================
Hệ thống huấn luyện multi-agent chơi bóng đá dùng thuật toán PPO.
"""

from .PPO import PPOAgent, PPOConfig, MultiAgentTrainer, ActorCriticNetwork

__all__ = [
    'PPOAgent',
    'PPOConfig',
    'MultiAgentTrainer',
    'ActorCriticNetwork',
]
