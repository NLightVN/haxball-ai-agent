"""
PPO.py - Thuật toán PPO cho Multi-Agent Learning
==================================================
Lớp Actor-Critic Network, PPO Agent, và Multi-Agent Trainer.
Phần của hệ thống huấn luyện multi-agent agents chơi bóng đá.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

class ActorCriticNetwork(nn.Module):
    """
    Mạng Actor-Critic cho một agent.
    
    Cấu trúc:
    - Backbone chia sẻ: MLP layers trích feature
    - Actor head: output hành động (9 lựa chọn)
    - Kick head: output quyết định sút bóng (2 lựa chọn)
    - Critic head: output giá trị state (value function)
    """
    
    def __init__(self, obs_dim: int, action_dim: int = 9, hidden_dim: int = 128):
        """
        Khởi tạo mạng Actor-Critic.
        
        Tham số:
            obs_dim: Kích thước observation input
            action_dim: Số lựa chọn hành động (9 hướng di chuyển)
            hidden_dim: Kích thước hidden layer
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Kick action head (binary: kick or not)
        self.kick_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tính toán forward qua mạng.
        
        Vào: obs từ environment
        Ra:
            action_logits: xác suất của 9 hành động di chuyển
            kick_logits: xác suất sút bóng hay không
            value: ước lượng giá trị state
        """
        # Trích feature từ backbone
        features = self.backbone(obs)
        
        # Tính output 3 đầu
        action_logits = self.actor_head(features)  # 9 hành động
        kick_logits = self.kick_head(features)      # sút hay không
        value = self.critic_head(features)          # giá trị state
        
        return action_logits, kick_logits, value


# ─────────────────────────────────────────────────────────────────────────────
# PPO Algorithm
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """Cài đặt hyperparameter cho huấn luyện PPO."""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    epsilon: float = 0.2  # Clip range
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    """
    Một agent sử dụng thuật toán PPO.
    
    Học policy (quy luật chọn hành động) để maximize reward.
    Dùng value function để ước lượng giá trị state.
    """
    
    def __init__(self, obs_dim: int, action_dim: int = 9, config: Optional[PPOConfig] = None):
        """
        Khởi tạo PPO agent.
        
        Tham số:
            obs_dim: Kích thước observation
            action_dim: Số lựa chọn hành động
            config: Cài đặt hyperparameter PPO
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        
        # Network
        self.network = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Storage cho trajectory
        self.trajectory: Dict[str, List] = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': [],
        }
    
    def select_action(self, obs: np.ndarray, training: bool = True) -> Tuple[int, int, float]:
        """
        Chọn hành động dựa trên policy hiện tại.
        
        Vào:
            obs: observation từ env
            training: True = chọn random theo xác suất, False = chọn tốt nhất
            
        Ra:
            movement_action: hướng di chuyển (0-8)
            kick_action: sút bóng (0-1)
            log_prob: xác suất log của hành động được chọn
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, kick_logits, value = self.network(obs_tensor)
        
        # Tính xác suất hành động di chuyển (9 hướng)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if training:
            # Chế độ huấn luyện: sample từ distribution
            action_dist = Categorical(action_probs)
            movement_action = action_dist.sample().item()
            action_log_prob = action_dist.log_prob(torch.tensor(movement_action)).item()
        else:
            # Chế độ đánh giá: chọn hành động tốt nhất
            movement_action = action_probs.argmax(dim=-1).item()
            action_log_prob = torch.log(action_probs[0, movement_action]).item()
        
        # Tính xác suất quyết định sút bóng
        kick_probs = F.softmax(kick_logits, dim=-1)
        
        if training:
            kick_dist = Categorical(kick_probs)
            kick_action = kick_dist.sample().item()
            kick_log_prob = kick_dist.log_prob(torch.tensor(kick_action)).item()
        else:
            kick_action = kick_probs.argmax(dim=-1).item()
            kick_log_prob = torch.log(kick_probs[0, kick_action]).item()
        
        # Total log prob
        log_prob = action_log_prob + kick_log_prob
        
        value_est = value.item()
        
        return movement_action, kick_action, log_prob
    
    def record_step(self, obs: np.ndarray, action: Tuple[int, int],
                   reward: float, done: bool, next_value: float = 0.0):
        """
        Lưu một bước vào buffer quá trình.
        
        Vào:
            obs: observation
            action: (hành động di chuyển, hành động sút)
            reward: phần thưởng nhận được
            done: quá trình kết thúc?
            next_value: giá trị next state (dùng cho GAE)
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, kick_logits, value = self.network(obs_tensor)
            
            # Re-compute log prob
            action_probs = F.softmax(action_logits, dim=-1)
            kick_probs = F.softmax(kick_logits, dim=-1)
            
            action_dist = Categorical(action_probs)
            kick_dist = Categorical(kick_probs)
            
            log_prob = (action_dist.log_prob(torch.tensor(action[0])) +
                       kick_dist.log_prob(torch.tensor(action[1]))).item()
        
        self.trajectory['obs'].append(obs)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['dones'].append(done)
        self.trajectory['values'].append(value.item())
        self.trajectory['log_probs'].append(log_prob)
    
    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính advantage dùng GAE (Generalized Advantage Estimation).
        Được dùng để thay vì raw reward.
        
        Ra:
            advantages: lợi thế tương đối
            returns: target return cho value function
        """
        values = np.array(self.trajectory['values'])
        rewards = np.array(self.trajectory['rewards'])
        dones = np.array(self.trajectory['dones'])
        
        # Thêm giá trị terminal
        values = np.append(values, 0.0)
        
        advantages = []
        gae = 0
        
        # Tính GAE từ phải sang trái
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]
        
        # Chuẩn hóa advantage để train ổn định
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Cập nhật mạng dùng PPO loss.
        
        Tham số:
            advantages: GAE advantages
            returns: Target returns cho value function
        """
        obs_batch = np.array(self.trajectory['obs'])
        actions_batch = np.array(self.trajectory['actions'])
        old_log_probs = np.array(self.trajectory['log_probs'])
        
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        actions_tensor = torch.from_numpy(actions_batch).long().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        old_log_probs_tensor = torch.from_numpy(old_log_probs).float().to(self.device)
        
        # Lặp NUM_EPOCHS lần qua dữ liệu
        for _ in range(self.config.num_epochs):
            indices = np.arange(len(obs_batch))
            np.random.shuffle(indices)
            
            # Chia batch nhỏ để cập nhật tham số
            for start_idx in range(0, len(obs_batch), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(obs_batch))
                batch_indices = indices[start_idx:end_idx]
                
                # Tính output từ mạng
                action_logits, kick_logits, values = self.network(obs_tensor[batch_indices])
                
                # Tính xác suất hành động mới
                action_probs = F.softmax(action_logits, dim=-1)
                kick_probs = F.softmax(kick_logits, dim=-1)
                
                action_dists = Categorical(action_probs)
                kick_dists = Categorical(kick_probs)
                
                new_log_probs = (
                    action_dists.log_prob(actions_tensor[batch_indices, 0]) +
                    kick_dists.log_prob(actions_tensor[batch_indices, 1])
                )
                
                # Tính PPO loss (clipped policy loss)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor[batch_indices])
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = (torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) *
                        advantages_tensor[batch_indices])
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Loss của value function
                value_loss = F.mse_loss(values.squeeze(), returns_tensor[batch_indices])
                
                # Entropy bonus (khuyến khích khám phá)
                entropy = action_dists.entropy().mean() + kick_dists.entropy().mean()
                
                # Tính tổng loss
                total_loss = (policy_loss +
                             self.config.value_coef * value_loss -
                             self.config.entropy_coef * entropy)
                
                # Cập nhật tham số mạng
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
    
    def update(self):
        """Cập nhật mạng từ dữ liệu đã thu thập."""
        advantages, returns = self.compute_advantages()
        self.train_step(advantages, returns)
        self.clear_trajectory()
    
    def clear_trajectory(self):
        """Xóa buffer để chuẩn bị cho episode tiếp theo."""
        for key in self.trajectory:
            self.trajectory[key] = []
    
    def save_checkpoint(self, path: str):
        """Lưu checkpoint mô hình."""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Tải checkpoint mô hình."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Coordinator
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentTrainer:
    """
    Điều phối huấn luyện nhiều agents chơi bóng đá.
    Mỗi agent đều học policy của riêng mình độc lập.
    """
    
    def __init__(self, n_agents: int, obs_dim: int, config: Optional[PPOConfig] = None):
        """
        Khởi tạo multi-agent trainer.
        
        Tham số:
            n_agents: Số agents cần train
            obs_dim: Kích thước observation
            config: Cài đặt PPO dùng chung
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.config = config or PPOConfig()
        
        # Khởi tạo N agents
        self.agents: List[PPOAgent] = [
            PPOAgent(obs_dim, config=self.config)
            for _ in range(n_agents)
        ]
        
        # Thống kê huấn luyện
        self.total_steps = 0
        self.episode_rewards: List[float] = []
    
    def select_actions(self, observations: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chọn hành động cho tất cả agents.
        
        Vào:
            observations: observation của tất cả agents
            training: Chế độ train hay eval
            
        Ra:
            movement_actions: hành động di chuyển
            kick_actions: hành động sút
        """
        movement_actions = []
        kick_actions = []
        
        # Mỗi agent chọn action của riêng nó
        for i, agent in enumerate(self.agents):
            move, kick, _ = agent.select_action(observations[i], training)
            movement_actions.append(move)
            kick_actions.append(kick)
        
        return np.array(movement_actions), np.array(kick_actions)
    
    def record_step(self, observations: np.ndarray, actions: np.ndarray,
                   kick_actions: np.ndarray, rewards: np.ndarray,
                   dones: np.ndarray, next_observations: np.ndarray):
        """Lưu một bước cho tất cả agents."""
        # Mỗi agent ghi lại bước của nó
        for i, agent in enumerate(self.agents):
            agent.record_step(
                observations[i],
                (actions[i].item(), kick_actions[i].item()),
                rewards[i],
                dones[i] if isinstance(dones, np.ndarray) else dones,
                0.0
            )
        
        self.total_steps += 1
    
    def update_all(self):
        """Cập nhật mô hình của tất cả agents."""
        for agent in self.agents:
            agent.update()
    
    def save_checkpoint(self, directory: str):
        """Lưu checkpoint cho tất cả agents."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f'agent_{i}.pt')
            agent.save_checkpoint(path)
    
    def load_checkpoint(self, directory: str):
        """Tải checkpoint cho tất cả agents."""
        import os
        
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f'agent_{i}.pt')
            if os.path.exists(path):
                agent.load_checkpoint(path)
