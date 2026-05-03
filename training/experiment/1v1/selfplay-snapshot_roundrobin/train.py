"""
train.py - Self-Play with Snapshot Round-Robin for 1v1 HaxBall
==============================================================

Thuật toán:
  1. Learner (PPO) thi đấu round-robin với toàn bộ snapshot pool + chính mình (frozen).
  2. Sau mỗi tournament → 1 PPO update.
  3. Lịch lưu snapshot:
       - 250 updates đầu : mỗi 50 updates (tại 50, 100, 150, 200, 250)
       - Sau đó          : mỗi 250 updates
  4. Dense reward (chỉ 250 updates đầu):
       - Bóng ở nửa sân mình (flip*bx < 0) → -DENSE_COEF
       - Bóng đứng yên > 5s → phạt tăng dần cho đến khi player chạm bóng
  5. Sparse reward (toàn thời gian):
       - Ghi bàn      : +100 + 50 × (số bàn thắng đã ghi trước đó)
       - Bị ghi bàn   : -80  - 40 × (số bàn thắng đã ghi trước đó)
       - Hòa (hết giờ): 0
  6. End-of-match bonus:
       - Thắng : +500
       - Thua  : -500
  7. goal_limit mỗi episode được random trong [3, 4, 5, 6].
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys_paths = [
    str(_HERE.parent.parent.parent.parent),   # project root (haxball_env.py)
    str(_HERE.parent.parent.parent),          # training/ (PPO.py)
]
import sys
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from haxball_env import HaxballEnv, OBS_DIM
from PPO import PPOAgent, PPOConfig, ActorCriticNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Environment
    HW: float = 368.0
    HH: float = 171.0
    goal_y: float = 64.0
    time_limit_min: int = 3
    spawn_mode: str = "haxball"

    # Reward — sparse (per-goal, scaling)
    goal_scored_base: float   = 100.0   # reward khi ghi bàn (base)
    goal_scored_scale: float  = 50.0    # +scale × số bàn đã ghi trước đó
    goal_conceded_base: float = -80.0   # penalty khi bị ghi bàn (base)
    goal_conceded_scale: float = -40.0  # +scale × số bàn đã ghi trước đó

    # Reward — end-of-match bonus (flat)
    win_bonus: float   = 500.0    # bonus khi thắng
    lose_bonus: float  = -500.0   # penalty khi thua

    # Dense reward
    dense_coef: float  = 0.001      # reward/step khi bóng ở đúng nửa sân

    # Ball idle penalty (bóng đứng yên quá lâu)
    idle_grace_seconds: float  = 5.0    # bắt đầu phạt sau N giây bóng đứng yên
    idle_penalty_coef: float   = 0.000005  # hệ số phạt, tăng tuyến tính theo thời gian idle
    ball_speed_threshold: float = 0.3   # ngưỡng tốc độ bóng coi là "đứng yên"

    # Schedule
    dense_reward_phase: int        = 250   # updates đầu có dense reward
    snapshot_interval_early: int   = 50    # mỗi 50 updates trong 250 đầu
    snapshot_interval_late: int    = 250   # mỗi 250 updates sau đó

    # goal_limit range
    goal_limit_min: int = 3
    goal_limit_max: int = 6

    # Training
    max_updates: int  = 100_000
    log_interval: int = 10          # log mỗi N updates
    full_ckpt_interval: int = 500   # lưu full checkpoint mỗi N updates

    # Paths
    checkpoint_dir: str = str(_HERE / "checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Pool
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotPool:
    """
    Lưu danh sách các frozen snapshot (state_dict) của learner.
    Dùng temp_network để inference mà không cần tạo nhiều network object.
    """

    def __init__(self, obs_dim: int, device: torch.device) -> None:
        self.obs_dim    = obs_dim
        self.device     = device
        self.snapshots: List[Dict] = []

    def add(self, network: ActorCriticNetwork) -> int:
        """Deep-copy state_dict của network và thêm vào pool."""
        snap = copy.deepcopy(network.state_dict())
        self.snapshots.append(snap)
        idx = len(self.snapshots) - 1
        logging.info(f"[SnapshotPool] Snapshot #{idx} added. Pool size: {len(self.snapshots)}")
        return idx

    def get_action(self, snap_idx: int,
                   obs: np.ndarray,
                   temp_net: ActorCriticNetwork) -> Tuple[int, int]:
        """Inference từ snapshot snap_idx (không gradient)."""
        temp_net.load_state_dict(self.snapshots[snap_idx])
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            a_logits, k_logits, _ = temp_net(obs_t)
        move = int(torch.argmax(a_logits, dim=-1).item())
        kick = int(torch.argmax(k_logits, dim=-1).item())
        return move, kick

    def __len__(self) -> int:
        return len(self.snapshots)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shaper
# ─────────────────────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Dense reward (chỉ trong dense_phase đầu):
      obs[8] = flip * bx / NORM
      < 0 → bóng ở nửa sân mình → -dense_coef

    Sparse reward — per-goal (scaling theo số bàn đã ghi trước đó):
      Ghi bàn   : +goal_scored_base  + goal_scored_scale  × (learner_goals trước đó)
      Bị ghi bàn: +goal_conceded_base + goal_conceded_scale × (learner_goals trước đó)

    End-of-match bonus (khi trận kết thúc do đạt goal_limit hoặc hết giờ):
      Thắng: +win_bonus
      Thua : +lose_bonus
      Hòa  : 0

    Ball idle penalty (toàn thời gian):
      Bóng đứng yên > idle_grace_seconds → -idle_penalty_coef × (steps vượt grace)
      Reset khi bóng di chuyển trở lại (player chạm bóng).
    """

    BALL_X_OBS_IDX = 8  # flip * bx / NORM trong obs vector

    def __init__(self, cfg: TrainConfig) -> None:
        self.goal_scored_base    = cfg.goal_scored_base
        self.goal_scored_scale   = cfg.goal_scored_scale
        self.goal_conceded_base  = cfg.goal_conceded_base
        self.goal_conceded_scale = cfg.goal_conceded_scale
        self.win_bonus           = cfg.win_bonus
        self.lose_bonus          = cfg.lose_bonus
        self.idle_penalty_coef   = cfg.idle_penalty_coef
        self.idle_grace_steps    = int(cfg.idle_grace_seconds * 60 / 3)  # ticks→env steps
        self.ball_speed_threshold = cfg.ball_speed_threshold
        self.dense_coef          = cfg.dense_coef
        self.dense_phase         = cfg.dense_reward_phase

    def step_reward(self, obs_learner: np.ndarray, update_count: int) -> float:
        """Dense shaping (chỉ trong dense_phase updates đầu). Chỉ phạt khi bóng ở nửa sân mình."""
        if update_count >= self.dense_phase:
            return 0.0
        ball_x_flipped = float(obs_learner[self.BALL_X_OBS_IDX])
        return -self.dense_coef if ball_x_flipped < 0.0 else 0.0

    def goal_reward(self, goal_result: int, learner_goals_before: int) -> float:
        """
        Sparse reward khi có bàn thắng, từ góc nhìn của learner (luôn là RED).
          goal_result == +1 → RED ghi bàn → learner ghi bàn
            reward = goal_scored_base + goal_scored_scale × learner_goals_before
          goal_result == -1 → BLUE ghi bàn → learner bị ghi bàn
            reward = goal_conceded_base + goal_conceded_scale × learner_goals_before
        """
        if goal_result == 1:
            return self.goal_scored_base + self.goal_scored_scale * learner_goals_before
        if goal_result == -1:
            return self.goal_conceded_base + self.goal_conceded_scale * learner_goals_before
        return 0.0

    def match_end_reward(self, learner_score: int, opponent_score: int) -> float:
        """
        Bonus khi trận kết thúc (terminated hoặc truncated).
          Thắng: +win_bonus
          Thua : +lose_bonus
          Hòa  : 0
        """
        if learner_score > opponent_score:
            return self.win_bonus
        if learner_score < opponent_score:
            return self.lose_bonus
        return 0.0

    def idle_penalty(self, ball_idle_steps: int) -> float:
        """
        Phạt tăng dần khi bóng đứng yên quá lâu.
        Chỉ phạt sau idle_grace_steps, tăng tuyến tính.
        """
        if ball_idle_steps <= self.idle_grace_steps:
            return 0.0
        excess = ball_idle_steps - self.idle_grace_steps
        return -self.idle_penalty_coef * excess

# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: HaxballEnv,
    learner: PPOAgent,
    snapshot_pool: SnapshotPool,
    temp_net: ActorCriticNetwork,
    opp_snap_idx: Optional[int],   # None = self-play (frozen copy of learner)
    update_count: int,
    reward_shaper: RewardShaper,
    goal_limit: int,
) -> Dict[str, List]:
    """
    Chạy 1 episode. Learner = RED (obs index 0), Opponent = BLUE (obs index 1).

    Nếu opp_snap_idx is None → thi đấu với frozen copy của learner hiện tại.

    Trả về trajectory dict (chỉ cho learner).
    """
    # Override goal_limit cho episode này
    env.goal_limit = goal_limit

    obs_list, _ = env.reset()
    learner_obs: np.ndarray = obs_list[0]
    opp_obs: np.ndarray     = obs_list[1]

    # Frozen copy nếu vs self
    if opp_snap_idx is None:
        frozen_state = copy.deepcopy(learner.network.state_dict())
        temp_net.load_state_dict(frozen_state)

    traj: Dict[str, List] = {
        k: [] for k in ("obs", "actions", "rewards", "dones", "values", "log_probs")
    }
    done = False

    # Track goals scored by learner (RED) during this episode
    learner_goals = 0

    # Track ball idle time (bóng đứng yên)
    ball_idle_steps = 0

    while not done:
        # ── Learner action ──────────────────────────────────────────────────
        move_l, kick_l, log_prob_l = learner.select_action(learner_obs, training=True)

        # ── Opponent action (frozen / no grad) ─────────────────────────────
        if opp_snap_idx is None:
            opp_obs_t = torch.from_numpy(opp_obs).float().unsqueeze(0).to(learner.device)
            with torch.no_grad():
                a_logits, k_logits, _ = temp_net(opp_obs_t)
            move_o = int(torch.argmax(a_logits, dim=-1).item())
            kick_o = int(torch.argmax(k_logits, dim=-1).item())
        else:
            move_o, kick_o = snapshot_pool.get_action(opp_snap_idx, opp_obs, temp_net)

        # ── Env step ────────────────────────────────────────────────────────
        actions = [[move_l, kick_l], [move_o, kick_o]]
        obs_list, _, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        goal_result = info.get("goal", 0)

        # ── Ball idle tracking ──────────────────────────────────────────────
        ball_speed = (env.ball.xs ** 2 + env.ball.ys ** 2) ** 0.5
        if ball_speed < reward_shaper.ball_speed_threshold:
            ball_idle_steps += 1
        else:
            ball_idle_steps = 0
        # Ghi bàn cũng reset idle (bóng sẽ spawn lại)
        if goal_result != 0:
            ball_idle_steps = 0

        # ── Reward ──────────────────────────────────────────────────────────
        dense_r = reward_shaper.step_reward(learner_obs, update_count)

        # Per-goal sparse reward (scaling theo số bàn đã ghi trước đó)
        goal_r = 0.0
        if goal_result != 0:
            goal_r = reward_shaper.goal_reward(goal_result, learner_goals)
            # Cập nhật số bàn learner đã ghi SAU khi tính reward
            if goal_result == 1:
                learner_goals += 1

        # End-of-match bonus (chỉ khi trận kết thúc)
        match_end_r = 0.0
        if done:
            match_end_r = reward_shaper.match_end_reward(
                env.red_score, env.blue_score
            )

        # Ball idle penalty (bóng đứng yên quá lâu)
        idle_r = reward_shaper.idle_penalty(ball_idle_steps)

        total_r = dense_r + goal_r + match_end_r + idle_r

        # ── Value estimate ───────────────────────────────────────────────────
        obs_t = torch.from_numpy(learner_obs).float().unsqueeze(0).to(learner.device)
        with torch.no_grad():
            _, _, val_t = learner.network(obs_t)
        value_est = float(val_t.item())

        # ── Store ────────────────────────────────────────────────────────────
        traj["obs"].append(learner_obs.copy())
        traj["actions"].append((move_l, kick_l))
        traj["rewards"].append(total_r)
        traj["dones"].append(done)
        traj["values"].append(value_est)
        traj["log_probs"].append(log_prob_l)

        learner_obs = obs_list[0]
        opp_obs     = obs_list[1]

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Tournament (Round-Robin)
# ─────────────────────────────────────────────────────────────────────────────

def run_tournament(
    env: HaxballEnv,
    learner: PPOAgent,
    snapshot_pool: SnapshotPool,
    temp_net: ActorCriticNetwork,
    update_count: int,
    reward_shaper: RewardShaper,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Round-robin: learner vs mỗi snapshot + vs chính nó (None).
    Gộp toàn bộ trajectory vào learner.trajectory (để PPO update).
    Trả về stats.
    """
    opponents: List[Optional[int]] = list(range(len(snapshot_pool))) + [None]

    stats: Dict[str, Any] = {
        "n_matches":   len(opponents),
        "wins":        0,
        "losses":      0,
        "draws":       0,
        "total_steps": 0,
    }

    # Xóa trajectory cũ
    learner.clear_trajectory()

    for opp_idx in opponents:
        goal_limit = random.randint(cfg.goal_limit_min, cfg.goal_limit_max)

        ep_traj = run_episode(
            env, learner, snapshot_pool, temp_net,
            opp_snap_idx=opp_idx,
            update_count=update_count,
            reward_shaper=reward_shaper,
            goal_limit=goal_limit,
        )

        # Append vào learner.trajectory
        for key in learner.trajectory:
            learner.trajectory[key].extend(ep_traj[key])

        # ── Thống kê (dùng red_score / blue_score sau episode) ──
        red_s  = env.red_score
        blue_s = env.blue_score
        if red_s > blue_s:
            stats["wins"]   += 1
        elif red_s < blue_s:
            stats["losses"] += 1
        else:
            stats["draws"]  += 1

        stats["total_steps"] += len(ep_traj["rewards"])

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Schedule
# ─────────────────────────────────────────────────────────────────────────────

def should_save_snapshot(update_count: int, cfg: TrainConfig) -> bool:
    """
    Lịch lưu snapshot:
      - Trong [1 .. dense_reward_phase]: mỗi snapshot_interval_early updates
      - Sau dense_reward_phase          : mỗi snapshot_interval_late updates
    """
    if update_count <= 0:
        return False
    if update_count <= cfg.dense_reward_phase:
        return update_count % cfg.snapshot_interval_early == 0
    return update_count % cfg.snapshot_interval_late == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(_HERE / "train.log"), mode="w"),
        ],
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env = HaxballEnv(
        n_per_team=1,
        HW=cfg.HW, HH=cfg.HH, goal_y=cfg.goal_y,
        spawn_mode=cfg.spawn_mode,
        time_limit_min=cfg.time_limit_min,
        goal_limit=3,   # sẽ bị override mỗi episode
    )

    # ── Learner ───────────────────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        num_epochs=4,
        batch_size=64,
    )
    learner = PPOAgent(obs_dim=OBS_DIM, action_dim=9, config=ppo_cfg)

    # ── Temp network (dùng chung để inference snapshot, không train) ──────────
    temp_net = ActorCriticNetwork(obs_dim=OBS_DIM, action_dim=9).to(learner.device)
    temp_net.eval()

    # ── Snapshot pool & reward shaper ─────────────────────────────────────────
    snapshot_pool = SnapshotPool(obs_dim=OBS_DIM, device=learner.device)
    reward_shaper = RewardShaper(cfg)

    # Snapshot đầu tiên = random init
    snapshot_pool.add(learner.network)
    logging.info(f"OBS_DIM={OBS_DIM} | Device={learner.device}")
    logging.info("Initial snapshot added (random init). Training begins.")

    update_count = 0
    total_steps  = 0
    t_start      = time.time()

    while update_count < cfg.max_updates:
        # ── Tournament ────────────────────────────────────────────────────────
        stats = run_tournament(
            env, learner, snapshot_pool, temp_net,
            update_count=update_count,
            reward_shaper=reward_shaper,
            cfg=cfg,
        )
        total_steps += stats["total_steps"]

        # ── PPO Update ────────────────────────────────────────────────────────
        learner.update()
        update_count += 1

        # ── Snapshot ──────────────────────────────────────────────────────────
        if should_save_snapshot(update_count, cfg):
            snap_idx  = snapshot_pool.add(learner.network)
            snap_path = os.path.join(cfg.checkpoint_dir, f"snapshot_{update_count:06d}.pt")
            torch.save(learner.network.state_dict(), snap_path)
            logging.info(f"Snapshot #{snap_idx} saved → {snap_path}")

        # ── Full checkpoint ───────────────────────────────────────────────────
        if update_count % cfg.full_ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"learner_{update_count:06d}.pt")
            learner.save_checkpoint(ckpt_path)
            logging.info(f"Full checkpoint saved → {ckpt_path}")

        # ── Log ───────────────────────────────────────────────────────────────
        if update_count % cfg.log_interval == 0:
            phase   = "DENSE+SPARSE" if update_count <= cfg.dense_reward_phase else "SPARSE"
            elapsed = time.time() - t_start
            win_rate = stats["wins"] / max(1, stats["n_matches"])
            logging.info(
                f"Update {update_count:5d} | {phase} | "
                f"pool={len(snapshot_pool):3d} | "
                f"matches={stats['n_matches']} "
                f"W={stats['wins']} L={stats['losses']} D={stats['draws']} "
                f"wr={win_rate:.0%} | "
                f"steps/tour={stats['total_steps']:4d} | "
                f"total={total_steps:,} | {elapsed:.0f}s"
            )

    # ── Final ─────────────────────────────────────────────────────────────────
    learner.save_checkpoint(os.path.join(cfg.checkpoint_dir, "learner_final.pt"))
    logging.info("Training complete.")


if __name__ == "__main__":
    train()
