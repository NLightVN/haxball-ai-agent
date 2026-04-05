"""
1v1 - elo package

Hệ thống huấn luyện Elo-based curriculum cho 1v1 HaxBall agents.
"""

from .train import (
    EloBasedCurriculum,
    AgentPool,
    AgentStats,
    k_factor,
    simulate_match,
    process_match_result,
)

__all__ = [
    'EloBasedCurriculum',
    'AgentPool',
    'AgentStats',
    'k_factor',
    'simulate_match',
    'process_match_result',
]
