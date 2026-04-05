"""
train.py - Elo-Based Curriculum Training for 1v1 HaxBall Agents
================================================================

Hệ thống huấn luyện multi-agent với curriculum dựa trên Elo rating:

1. **Quản lý Agent Pool**: Bắt đầu 10 agents, thêm 10 agents mới sau mỗi chu kỳ
   - 10 agents  × 10 rounds = 50 matches (5 cặp × 10)
   - 20 agents  × 20 rounds = 200 matches (10 cặp × 20)
   - ...
   - 100 agents × 100 rounds = 5000 match (50 cặp × 100)

2. **Chọn Đối Thủ bằng Bell Curve**: 
   - Mỗi agent chọn đối thủ theo Gaussian distribution với mean = agent's Elo
   - Tạo n/2 cặp agents

3. **Huấn Luyện Song Song**:
   - Chạy tất cả matches của cặp hiện tại song song
   - Update weights + Elo cho tất cả agents
   - Tái tạo cặp cho round tiếp theo

4. **Elo Update**:
   - K-factor phụ thuộc vào games_played và pool_size
   - Pool nhỏ → K thấp, K lớn khi pool to và ít matches
"""

import os
import sys
import json
import math
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from scipy.stats import norm
import torch

# Thêm paths để import từ training/
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from observations import ObservationProcessor
from PPO import PPOAgent, PPOConfig


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_ELO = 1000.0  # Elo khởi đầu cho agents mới
ELO_K_BASE = 64.0     # K-factor cơ bản


@dataclass
class EloConfig:
    """Cấu hình Elo rating system."""
    initial_elo: float = INITIAL_ELO
    k_base: float = ELO_K_BASE
    

def k_factor(games_played: int, pool_size: int) -> float:
    """
    Tính K-factor cho Elo update.
    
    K càng cao → Elo biến động nhiều hơn (tốt khi agent mới hoặc pool nhỏ)
    K càng thấp → Elo ổn định hơn (tốt khi pool to và đã chơi nhiều)
    
    Tham số:
        games_played: Số trận đã chơi của agent
        pool_size: Kích thước pool (số agents)
    
    Trả về:
        K-factor value
    """
    # Pool càng nhỏ → K càng thấp để tránh biến động lớn
    pool_factor = min(1.0, pool_size / 50.0)
    
    # Dùng base_k phụ thuộc vào games_played
    # Nhiều trận → K thấp hơn (elo ổn định)
    base_k = max(16.0, 64.0 / (1.0 + games_played / 20.0))
    
    return base_k * pool_factor


def dynamic_sigma(elos: np.ndarray) -> float:
    """
    Tính sigma động cho bell curve opponent selection dựa trên Elo spread.
    
    Sigma scale theo phạm vi Elo trong pool:
    - Càng nhiều agents với Elo khác nhau → sigma lớn → chọn opponent đa dạng
    - Pool đồng nhất → sigma nhỏ → chọn opponent gần level
    
    Tham số:
        elos: Array of Elo ratings in pool
    
    Trả về:
        Dynamic sigma value (tối thiểu 75 rating points)
    """
    if len(elos) < 2:
        return 75.0
    
    spread = float(np.max(elos) - np.min(elos))
    return max(75.0, spread * 0.15)


@dataclass
class AgentStats:
    """Thống kê của một agent trong pool."""
    agent_id: int
    elo: float = INITIAL_ELO
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    ppo_agent: Optional[PPOAgent] = field(default=None)
    
    def win_rate(self) -> float:
        """Tỷ lệ thắng."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def expected_score(self, opponent_elo: float) -> float:
        """
        Tính điểm kỳ vọng của agent so với đối thủ.
        
        Formula: E = 1 / (1 + 10^((opponent_elo - agent_elo) / 400))
        """
        return 1.0 / (1.0 + math.pow(10.0, (opponent_elo - self.elo) / 400.0))
    
    def update_elo(self, opponent_elo: float, actual_score: float, pool_size: int) -> float:
        """
        Update Elo rating.
        
        Tham số:
            opponent_elo: Elo của đối thủ
            actual_score: Kết quả (1.0 = win, 0.0 = loss, 0.5 = draw)
            pool_size: Kích thước pool để tính K-factor
        
        Trả về:
            Elo rating mới
        """
        expected = self.expected_score(opponent_elo)
        k = k_factor(self.games_played, pool_size)
        
        delta = k * (actual_score - expected)
        self.elo += delta
        self.games_played += 1
        
        if actual_score > 0.5:
            self.wins += 1
        elif actual_score < 0.5:
            self.losses += 1
        else:
            self.draws += 1
        
        return self.elo


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent Pool Management
# ═══════════════════════════════════════════════════════════════════════════════

class AgentPool:
    """Quản lý pool agents với Elo-based selection."""
    
    def __init__(self, initial_size: int = 10, ppo_config: Optional[PPOConfig] = None):
        """
        Khởi tạo agent pool.
        
        Tham số:
            initial_size: Số agents khởi đầu
            ppo_config: Cấu hình PPO cho các agents
        """
        self.agents: Dict[int, AgentStats] = {}
        self.agent_counter = 0
        self.ppo_config = ppo_config or PPOConfig()
        
        # Khởi tạo agents ban đầu
        for _ in range(initial_size):
            self._add_agent()
    
    def _add_agent(self) -> int:
        """Thêm agent mới vào pool."""
        agent_id = self.agent_counter
        obs_dim = 112  # TODO: Lấy từ ObservationProcessor.flat_obs_dim
        ppo_agent = PPOAgent(obs_dim, action_dim=9, config=self.ppo_config)
        
        stats = AgentStats(
            agent_id=agent_id,
            ppo_agent=ppo_agent
        )
        self.agents[agent_id] = stats
        self.agent_counter += 1
        
        logger.info(f"Agent {agent_id} added. Pool size: {len(self.agents)}")
        return agent_id
    
    def add_agents(self, count: int) -> List[int]:
        """Thêm nhiều agents mới."""
        new_ids = [self._add_agent() for _ in range(count)]
        return new_ids
    
    def get_sorted_agents(self) -> List[Tuple[int, float]]:
        """
        Lấy danh sách agents sắp xếp theo Elo từ thấp đến cao.
        
        Trả về: List[(agent_id, elo)]
        """
        agents_list = [(aid, stats.elo) for aid, stats in self.agents.items()]
        agents_list.sort(key=lambda x: x[1])  # Sort by Elo ascending
        return agents_list
    
    def select_opponent(self, agent_elo: float, exclude_id: Optional[int] = None) -> int:
        """
        Chọn đối thủ cho agent dựa trên bell curve distribution.
        
        Bell curve có mean = agent_elo, và sigma = dynamic_sigma(pool.elos).
        Sigma tự động scale theo phạm vi Elo trong pool.
        
        Tham số:
            agent_elo: Elo của agent cần tìm đối thủ
            exclude_id: ID agent cần loại trừ (VD: chính agent đó)
        
        Trả về:
            ID của agent được chọn
        """
        # Lấy danh sách ID + Elo tương ứng
        available_agents = []
        for aid, stats in self.agents.items():
            if exclude_id is None or aid != exclude_id:
                available_agents.append((aid, stats.elo))
        
        if not available_agents:
            raise ValueError("Không có đối thủ khả dụng!")
        
        # Tính sigma động dựa trên Elo spread
        elos = np.array([e for _, e in available_agents])
        sigma = dynamic_sigma(elos)
        
        # Sinh Elo cho đối thủ từ Gaussian: mean=agent_elo, std=sigma
        target_elo = np.random.normal(agent_elo, sigma)
        target_elo = np.clip(target_elo, elos.min() - 100, elos.max() + 100)
        
        # Tìm agent có Elo gần nhất với target_elo
        best_aid = min(available_agents, key=lambda x: abs(x[1] - target_elo))[0]
        
        return best_aid
    
    def create_pairs(self) -> List[Tuple[int, int]]:
        """
        Tạo n/2 cặp agents từ pool.
        
        Algorithm: Sắp xếp theo Elo từ thấp đến cao, sau đó mỗi agent chọn đối thủ
        theo bell curve với sigma động, tạo thành các cặp không trùng lặp.
        
        Trả về: List[(agent_id_1, agent_id_2)]
        """
        sorted_agents = self.get_sorted_agents()
        agent_ids = [aid for aid, _ in sorted_agents]
        
        pairs = []
        used = set()
        
        for agent_id in agent_ids:
            if agent_id in used:
                continue
            
            # Chọn đối thủ cho agent này (sigma động)
            opponent_id = self.select_opponent(
                self.agents[agent_id].elo,
                exclude_id=agent_id,
            )
            
            # Nếu opponent đã được sử dụng, chọn lại
            if opponent_id in used:
                # Thử lại (sigma vẫn động)
                for _ in range(5):
                    opponent_id = self.select_opponent(
                        self.agents[agent_id].elo,
                        exclude_id=agent_id,
                    )
                    if opponent_id not in used:
                        break
            
            if opponent_id not in used and agent_id != opponent_id:
                pairs.append((agent_id, opponent_id))
                used.add(agent_id)
                used.add(opponent_id)
        
        logger.info(f"Created {len(pairs)} pairs from {len(self.agents)} agents")
        return pairs
    
    def get_agent(self, agent_id: int) -> AgentStats:
        """Lấy thống kê của một agent."""
        return self.agents[agent_id]
    
    def get_agent_ppo(self, agent_id: int) -> PPOAgent:
        """Lấy PPO agent object."""
        return self.agents[agent_id].ppo_agent


# ═══════════════════════════════════════════════════════════════════════════════
#  Match Simulation (Placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_match(red_agent_id: int, blue_agent_id: int, pool: AgentPool,
                  num_episodes: int = 5, render: bool = False) -> Dict[str, Any]:
    """
    Mô phỏng một trận đấu giữa hai agents.
    
    NOTE: Đây là placeholder. Trong thực tế, cần kết nối với HaxballEnv.
    
    Tham số:
        red_agent_id: ID của agent RED
        blue_agent_id: ID của agent BLUE
        pool: AgentPool
        num_episodes: Số episodes mô phỏng
        render: Có render hay không
    
    Trả về:
        Dict với thông tin kết quả
    """
    red_stats = pool.get_agent(red_agent_id)
    blue_stats = pool.get_agent(blue_agent_id)
    
    # Mô phỏng kết quả (trong thực tế, cần chạy env thực):
    # - Tính xác suất thắng dựa trên Elo
    # - Thực chạy match
    # - Cập nhật trajectory của agents
    
    red_expected = red_stats.expected_score(blue_stats.elo)
    winner = 'red' if np.random.random() < red_expected else 'blue'
    
    result = {
        'red_id': red_agent_id,
        'blue_id': blue_agent_id,
        'winner': winner,
        'red_expected': red_expected,
        'blue_expected': 1.0 - red_expected,
        'episodes': num_episodes,
        'red_reward': 1.0 if winner == 'red' else -1.0,
        'blue_reward': 1.0 if winner == 'blue' else -1.0,
    }
    
    return result


def process_match_result(result: Dict[str, Any], pool: AgentPool) -> None:
    """
    Xử lý kết quả một trận đấu: update Elo + weights.
    
    Tham số:
        result: Kết quả trận đấu từ simulate_match()
        pool: AgentPool
    """
    red_id = result['red_id']
    blue_id = result['blue_id']
    pool_size = len(pool.agents)
    
    # Xác định score dựa trên kết quả
    if result['winner'] == 'red':
        red_score = 1.0
        blue_score = 0.0
    else:
        red_score = 0.0
        blue_score = 1.0
    
    # Update Elo
    red_stats = pool.get_agent(red_id)
    blue_stats = pool.get_agent(blue_id)
    
    red_stats.update_elo(blue_stats.elo, red_score, pool_size)
    blue_stats.update_elo(red_stats.elo, blue_score, pool_size)
    
    logger.info(
        f"Match: Agent {red_id} ({red_stats.elo:.0f}) vs Agent {blue_id} ({blue_stats.elo:.0f}) "
        f"→ Winner: {result['winner'].upper()}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Curriculum
# ═══════════════════════════════════════════════════════════════════════════════

class EloBasedCurriculum:
    """Hệ thống curriculum dựa trên Elo."""
    
    def __init__(self, initial_pool_size: int = 10, add_per_cycle: int = 10,
                 max_pool_size: int = 100, ppo_config: Optional[PPOConfig] = None):
        """
        Khởi tạo curriculum.
        
        Tham số:
            initial_pool_size: Số agents khởi đầu
            add_per_cycle: Số agents thêm mỗi cycle
            max_pool_size: Kích thước pool tối đa
            ppo_config: Cấu hình PPO
        """
        self.pool = AgentPool(initial_pool_size, ppo_config)
        self.add_per_cycle = add_per_cycle
        self.max_pool_size = max_pool_size
        
        self.cycle = 0
        self.total_matches = 0
        self.training_history = []
    
    def _run_one_cycle(self, matches_per_pair: int = 10) -> Dict[str, Any]:
        """
        Chạy một cycle của curriculum.
        
        Algorithm:
        1. Tạo pairs từ agents hiện tại
        2. Chạy matches_per_pair trận cho mỗi pair
        3. Update Elo + weights cho tất cả agents
        4. Thêm agents mới nếu pool_size < max_pool_size
        
        Trả về:
            Dict với thông tin về cycle
        """
        cycle_info = {
            'cycle': self.cycle,
            'pool_size': len(self.pool.agents),
            'matches_per_pair': matches_per_pair,
            'total_matches': 0,
            'results': [],
            'agent_stats': {},
        }
        
        # Tạo pairs
        pairs = self.pool.create_pairs()
        
        # Chạy matches
        for red_id, blue_id in pairs:
            for match_idx in range(matches_per_pair):
                result = simulate_match(red_id, blue_id, self.pool)
                process_match_result(result, self.pool)
                
                cycle_info['results'].append(result)
                cycle_info['total_matches'] += 1
        
        # Lưu thống kê agents
        for agent_id, stats in self.pool.agents.items():
            cycle_info['agent_stats'][agent_id] = {
                'elo': stats.elo,
                'games_played': stats.games_played,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': stats.win_rate(),
            }
        
        self.total_matches += cycle_info['total_matches']
        self.training_history.append(cycle_info)
        
        logger.info(
            f"Cycle {self.cycle} completed: {cycle_info['total_matches']} matches, "
            f"pool_size={cycle_info['pool_size']}"
        )
        
        return cycle_info
    
    def train(self, num_cycles: int = 10) -> List[Dict[str, Any]]:
        """
        Chạy training loop.
        
        Curriculum:
        - Cycle 0-1: 10 agents × 10 rounds = 50 matches
        - Cycle 2-3: 20 agents × 20 rounds = 200 matches
        - ...
        - Cycle 8-9: 100 agents × 100 rounds = 5000 matches
        
        Tham số:
            num_cycles: Số cycles để chạy
        
        Trả về:
            List của cycle info dicts
        """
        for _ in range(num_cycles):
            # Tính số matches của cycle này
            current_size = len(self.pool.agents)
            matches_per_pair = current_size
            
            # Chạy cycle
            cycle_info = self._run_one_cycle(matches_per_pair)
            
            # Thêm agents mới nếu có thể
            if len(self.pool.agents) < self.max_pool_size:
                new_ids = self.pool.add_agents(self.add_per_cycle)
                logger.info(f"Added {len(new_ids)} new agents. Pool size: {len(self.pool.agents)}")
            
            self.cycle += 1
        
        return self.training_history
    
    def get_top_agents(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Lấy top K agents theo Elo."""
        agents_list = [(aid, stats.elo) for aid, stats in self.pool.agents.items()]
        agents_list.sort(key=lambda x: x[1], reverse=True)
        return agents_list[:top_k]
    
    def save_checkpoint(self, path: str) -> None:
        """Lưu checkpoint của training."""
        checkpoint = {
            'cycle': self.cycle,
            'total_matches': self.total_matches,
            'pool_size': len(self.pool.agents),
            'agent_stats': {
                aid: {
                    'elo': stats.elo,
                    'games_played': stats.games_played,
                    'wins': stats.wins,
                    'losses': stats.losses,
                }
                for aid, stats in self.pool.agents.items()
            },
            'training_history': self.training_history,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint từ file."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.cycle = checkpoint['cycle']
        self.total_matches = checkpoint['total_matches']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main training loop."""
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    logger.info("Starting Elo-based Curriculum Training for 1v1 HaxBall")
    
    # Khởi tạo curriculum
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
    )
    
    curriculum = EloBasedCurriculum(
        initial_pool_size=10,
        add_per_cycle=10,
        max_pool_size=100,
        ppo_config=ppo_config
    )
    
    logger.info(f"Initial pool: {len(curriculum.pool.agents)} agents")
    
    # Chạy training
    try:
        history = curriculum.train(num_cycles=10)
        
        logger.info(f"Training completed: {curriculum.total_matches} total matches")
        logger.info(f"Pool size: {len(curriculum.pool.agents)} agents")
        
        # In top agents
        top_10 = curriculum.get_top_agents(10)
        logger.info("Top 10 agents by Elo:")
        for rank, (agent_id, elo) in enumerate(top_10, 1):
            stats = curriculum.pool.get_agent(agent_id)
            logger.info(
                f"  {rank}. Agent {agent_id}: {elo:.0f} "
                f"({stats.wins}W-{stats.losses}L, wr={stats.win_rate():.2%})"
            )
        
        # Lưu checkpoint
        checkpoint_path = Path(__file__).parent / 'checkpoints' / 'training_final.json'
        curriculum.save_checkpoint(str(checkpoint_path))
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        checkpoint_path = Path(__file__).parent / 'checkpoints' / 'training_interrupted.json'
        curriculum.save_checkpoint(str(checkpoint_path))
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
