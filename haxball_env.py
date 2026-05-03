"""
HaxballEnv — Multi-agent Haxball Physics Environment
=====================================================
Hỗ trợ:
  • n_per_team      : số cầu thủ mỗi bên (1, 2, 3, 4) → 1v1 / 2v2 / 3v3 / 4v4
  • HW, HH         : nửa chiều rộng / cao sân (px)  — tâm sân là gốc toạ độ
  • goal_y         : nửa cột cao cửa gôn (px)        — phần trên/dưới cửa = ±goal_y
                    ví dụ goal_y=85 → cửa gôn từ y=-85 đến y=+85
  • spawn_mode     :
      'haxball'  — các preset gốc HaxBall theo từng n_per_team
                   (mỗi nhóm spawn cụm lại ở sân nhà theo vòng tròn)
      'random'   — mỗi người random trên nửa sân của mình,
                   bóng spawn cách đều closest-RED và closest-BLUE
  • time_limit_min : giới hạn thời gian mỗi hiệp (phút, 1–6). Mặc định: 3.
                    max_steps = time_limit_min × 60s × 60Hz (tick thật).

Physics: port trung thành từ valn-v4 / test_index.html của haxball-agent-lite.

Action space mỗi agent : MultiDiscrete([9, 2])
    dim-0 : hướng di chuyển (0=đứng, 1=R, 2=L, 3=U, 4=D, 5=UR, 6=UL, 7=DR, 8=DL)
    dim-1 : sút bóng (0=không, 1=có)

Observation mỗi agent : float32 array (OBS_DIM,)
    Xem _get_obs_for() để biết chi tiết.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

 
# Physics constants (valn-v4.hbs)
 
BALL_R        = 5.8
BALL_DAMP     = 0.99
BALL_BCOEF    = 0.412
BALL_IMASS    = 1.5

PLYR_R        = 15.0
PLYR_DAMP     = 0.96
PLYR_IMASS    = 0.5
PLYR_BCOEF    = 0.5
PLYR_ACC      = 0.11
PLYR_KICK_ACC = 0.083
KICK_STR      = 4.545
KICK_RANGE    = 4.0   # surface-gap threshold: dist - r_player - r_ball < 4

POLE_R        = 4.0
POLE_BCOEF    = 0.1
POLE_IMASS    = 0.0   # immovable

OUTER_PAD     = 35.0  # players can go this far outside field lines

 
# Observation normalisation
NORM      = 800.0
MAX_SPEED = 10.0
DIAG      = math.sqrt(NORM ** 2 + NORM ** 2)  # ≈ 1131.4

N_TM_SLOTS  = 4   # max teammate slots in obs
N_OPP_SLOTS = 5   # max opponent slots in obs

# obs layout:
#   [0..3]   field constants   (4)
#   [4..7]   agent ↔ ball      (4)
#   [8..18]  dynamic state     (11)
#   [19..24] game state        (7) — time_remaining, possession, ready_to_shot, time_speed,
#                                    goal_step, my_goals/goal_limit, opp_goals/goal_limit
#   [25..64] teammates ×4     (10 each)
#   [65..114] opponents ×5    (10 each)
OBS_DIM = 4 + 4 + 11 + 7 + N_TM_SLOTS * 10 + N_OPP_SLOTS * 10  # = 116

 
# HaxBall default spawn positions (pixel, centre-field = 0,0)
# Gốc từ chuẩn HaxBall spawn vòng tròn mỗi team.
 
# Vị trí tương đối trên sân chuẩn HaxBall (HW=700, HH=320)
# Team RED spawn phía TRÁI  (x < 0), BLUE phải (x > 0)
# Các vị trí được scale theo HW/HH thực tế khi reset.
_HAXBALL_SPAWN_REL: dict[int, list[tuple[float, float]]] = {
    # n_per_team → [(rx, ry), ...]  trong đó rx,ry ∈ [-1,1] theo (HW, HH)
    1: [(-0.40,  0.00)],
    2: [(-0.45, -0.22),
        (-0.45,  0.22)],
    3: [(-0.50,  0.00),
        (-0.40, -0.30),
        (-0.40,  0.30)],
    4: [(-0.50, -0.20),
        (-0.50,  0.20),
        (-0.38, -0.40),
        (-0.38,  0.40)],
}

 
# Direction map (action dim-0)
 
DIR_MAP = np.array([
    [ 0,  0],  # 0 stay
    [ 1,  0],  # 1 right
    [-1,  0],  # 2 left
    [ 0, -1],  # 3 up
    [ 0,  1],  # 4 down
    [ 1, -1],  # 5 up-right
    [-1, -1],  # 6 up-left
    [ 1,  1],  # 7 down-right
    [-1,  1],  # 8 down-left
], dtype=np.float64)

FRAME_SKIP        = 3    # physics ticks per env step (RL training)
PHYSICS_HZ        = 60
DEFAULT_EP_S      = 15   # episode length in seconds (RL training shorthand)
DEFAULT_TIME_MIN  = 3    # default match time limit (minutes, play mode)


 
# Disc
 
class Disc:
    """Mutable physics disc — ball or player."""
    __slots__ = ['x', 'y', 'xs', 'ys', 'radius', 'imass', 'bcoef', 'damp']

    def __init__(self, x, y, xs, ys, radius, imass, bcoef, damp):
        self.x = float(x);  self.y  = float(y)
        self.xs = float(xs); self.ys = float(ys)
        self.radius = float(radius)
        self.imass  = float(imass)
        self.bcoef  = float(bcoef)
        self.damp   = float(damp)


 
# Disc-Disc collision (exact port from test_index.html)
 
def _resolve_dd(da: Disc, db: Disc) -> None:
    ddx = da.x - db.x; ddy = da.y - db.y
    dist = math.hypot(ddx, ddy)
    r_sum = da.radius + db.radius
    if 0 < dist <= r_sum:
        nx, ny = ddx / dist, ddy / dist
        imass_sum = da.imass + db.imass
        if imass_sum == 0:
            return
        mf      = da.imass / imass_sum
        overlap = r_sum - dist
        da.x += nx * overlap * mf;         da.y += ny * overlap * mf
        db.x -= nx * overlap * (1 - mf);   db.y -= ny * overlap * (1 - mf)
        rvn = (da.xs - db.xs) * nx + (da.ys - db.ys) * ny
        if rvn < 0:
            impulse = rvn * (da.bcoef * db.bcoef + 1)
            da.xs -= nx * impulse * mf;         da.ys -= ny * impulse * mf
            db.xs += nx * impulse * (1 - mf);   db.ys += ny * impulse * (1 - mf)


 
# Main Environment
 
class HaxballEnv(gym.Env):
    """
    Môi trường HaxBall đa agent, cấu hình linh hoạt.

    Parameters
    ----------
    n_per_team : int
        Số cầu thủ mỗi bên. Hỗ trợ 1, 2, 3, 4  (→ 1v1, 2v2, 3v3, 4v4).
    HW : float
        Nửa chiều rộng sân (px). Mặc định: 700 (chuẩn valn-v4).
    HH : float
        Nửa chiều cao sân (px). Mặc định: 320 (chuẩn valn-v4).
    goal_y : float
        Nửa chiều cao cửa gôn (px). Mặc định: 85 (chuẩn valn-v4).
    spawn_mode : str
        'haxball' — vị trí spawn HaxBall gốc (scale theo sân)
        'random'  — random mỗi nửa sân; bóng cách đều red và blue gần nhất
    time_limit_min : int
        Giới hạn thời gian mỗi hiệp (phút, 1–6). Mặc định: 3.
        max_steps = time_limit_min × 60 × 60  (ticks @ 60 Hz, không chia FRAME_SKIP).
        Dùng cho play mode; RL training thường override max_steps sau khi khởi tạo.
    ep_seconds : float or None
        Nếu truyền vào (không None), override time_limit_min.
        Backward compat cho RL training code dùng ep_seconds cũ.
    seed : int or None
        Seed cho RNG.

    Notes
    -----
    - Gốc toạ độ (0,0) là TÂM SÂN.
    - RED team spawn bên trái (x<0), goal của RED ở x=-HW.
    - BLUE team spawn bên phải (x>0), goal của BLUE ở x=+HW.
    - step() nhận **list** action độ dài n_per_team×2 (RED trước, BLUE sau).
    - _tick() chạy đúng 1 physics tick; dùng trực tiếp trong play mode.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        n_per_team     : int            = 1,
        HW             : float          = 700.0,
        HH             : float          = 320.0,
        goal_y         : float          = 85.0,
        spawn_mode     : str            = 'haxball',
        time_limit_min : int            = DEFAULT_TIME_MIN,
        ep_seconds     : Optional[float] = None,
        goal_limit     : int            = 3,
        seed           : Optional[int]  = None,
    ):
        super().__init__()

        #Validation 
        assert n_per_team in (1, 2, 3, 4), "n_per_team phải là 1, 2, 3, hoặc 4"
        assert HW > 0 and HH > 0,          "HW và HH phải dương"
        assert 0 < goal_y <= HH,           "goal_y phải trong khoảng (0, HH]"
        assert spawn_mode in ('haxball', 'random'), "spawn_mode phải là 'haxball' hoặc 'random'"
        assert 1 <= time_limit_min <= 6,   "time_limit_min phải trong [1, 6]"
        assert goal_limit >= 1,             "goal_limit phải >= 1"

        self.n_per_team     = n_per_team
        self.HW             = float(HW)
        self.HH             = float(HH)
        self.goal_y         = float(goal_y)
        self.spawn_mode     = spawn_mode
        self.time_limit_min = int(time_limit_min)
        self.goal_limit     = int(goal_limit)
        self._rng           = np.random.default_rng(seed)

        # max_steps: ticks @ 60 Hz (không chia FRAME_SKIP)
        # • Nếu ep_seconds được truyền → backward compat (RL training)
        # • Ngược lại → dùng time_limit_min
        if ep_seconds is not None:
            self.ep_seconds = float(ep_seconds)
            self.max_steps  = int(ep_seconds * PHYSICS_HZ)  # tick thật
        else:
            self.ep_seconds = float(time_limit_min * 60)
            self.max_steps  = time_limit_min * 60 * PHYSICS_HZ

        self.step_count = 0

        # Spaces
        single_action  = spaces.MultiDiscrete([9, 2])
        single_obs     = spaces.Box(low=-3.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32)
        n_total        = n_per_team * 2

        # Flatten thành Tuple spaces để tương thích SB3 VecEnv wrapper
        self.action_space      = spaces.Tuple([single_action] * n_total)
        self.observation_space = spaces.Tuple([single_obs]    * n_total)
        # Cũng expose single-agent version để dễ dùng
        self.single_action_space      = single_action
        self.single_observation_space = single_obs
        self.n_agents = n_total

        # State (populated at reset)
        self.ball:          Optional[Disc]    = None
        self.red_players:   list[Disc]        = []   # RED: team_id=1, x<0 side
        self.blue_players:  list[Disc]        = []   # BLUE: team_id=2, x>0 side
        self._poles:        list[Disc]        = []   # goal poles (rebuilt each reset)
        self.ready_to_shot_states: list[int]  = []   # Track ready_to_shot mode for each player
        self.red_score:     int               = 0    # Tỷ số đội RED
        self.blue_score:    int               = 0    # Tỷ số đội BLUE

    #  Public helpers 
    def set_field(self, HW: float, HH: float, goal_y: float) -> None:
        """Thay đổi kích thước sân và goal, có hiệu lực từ episode tiếp theo."""
        assert HW > 0 and HH > 0
        assert 0 < goal_y <= HH
        self.HW     = float(HW)
        self.HH     = float(HH)
        self.goal_y = float(goal_y)
        self._rebuild_poles()

    #  reset 
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        if self.spawn_mode == 'haxball':
            self._spawn_haxball()
        else:
            self._spawn_random()

        self._rebuild_poles()
        # Initialize ready_to_shot_states (0: idle, 1: ready_to_shot, 2: already_shot)
        self.ready_to_shot_states = [0] * self.n_agents
        self.red_score  = 0
        self.blue_score = 0
        obs = self._build_all_obs()
        return obs, {}

    #  step 
    def step(self, actions):
        """
        actions : list/tuple của n_agents action, mỗi action là [dir(0-8), kick(0/1)].
                  Thứ tự: [red_0, red_1, ..., blue_0, blue_1, ...]
        Returns
        -------
        obs, rewards, terminated, truncated, info
          obs        : list of n_agents obs arrays
          rewards    : list of n_agents float (hiện tại: +1 ghi bàn, -1 thủng lưới, 0 khác)
          terminated : bool
          truncated  : bool
          info       : dict
        """
        assert self.ball is not None, "Gọi reset() trước step()."
        actions = list(actions)

        goal_result = 0   # 0=none, 1=RED ghi bàn (vào Blue goal x=+HW), -1=BLUE ghi bàn
        for _ in range(FRAME_SKIP):
            r = self._tick(actions)
            if r != 0:
                goal_result = r
                break

        self.step_count += 1

        #  Rewards 
        n = self.n_per_team
        # Mỗi agent nhận reward theo team
        rewards = [0.0] * (n * 2)
        terminated = False

        if goal_result == 1:     # RED ghi bàn → Blue goal bên phải
            self.red_score += 1
            for i in range(n):
                rewards[i]     = +1.0   # RED thưởng
            for i in range(n, n * 2):
                rewards[i]     = -1.0   # BLUE phạt
            if self.red_score >= self.goal_limit:
                terminated = True
        elif goal_result == -1:  # BLUE ghi bàn → Red goal bên trái
            self.blue_score += 1
            for i in range(n):
                rewards[i]     = -1.0
            for i in range(n, n * 2):
                rewards[i]     = +1.0
            if self.blue_score >= self.goal_limit:
                terminated = True

        # Tự động đưa bóng về giữa sân nếu có bàn thắng và trận đấu chưa kết thúc
        if goal_result != 0 and not terminated:
            if self.spawn_mode == 'haxball':
                self._spawn_haxball()
            else:
                self._spawn_random()
            # Reset trạng thái kick sau khi có bàn thắng
            self.ready_to_shot_states = [0] * self.n_agents

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        obs = self._build_all_obs()
        info = {'goal': goal_result, 'step': self.step_count,
                'red_score': self.red_score, 'blue_score': self.blue_score}
        return obs, rewards, terminated, truncated, info

    #  Spawn helpers 
    def _spawn_haxball(self) -> None:
        """Spawn theo vị trí HaxBall gốc, scale theo (HW, HH)."""
        rel_positions = _HAXBALL_SPAWN_REL[self.n_per_team]
        HW, HH = self.HW, self.HH

        self.red_players  = []
        self.blue_players = []

        for rx, ry in rel_positions:
            # RED: bên trái (rx < 0) → giữ nguyên dấu
            self.red_players.append(
                Disc(rx * HW, ry * HH, 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            )
            # BLUE: đối xứng qua trục Y
            self.blue_players.append(
                Disc(-rx * HW, ry * HH, 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            )

        # Bóng ở giữa sân
        self.ball = Disc(0.0, 0.0, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

    def _spawn_random(self) -> None:
        """
        Mỗi cầu thủ random trên nửa sân của mình (cách tường ≥ PLYR_R).
        Sau đó bóng được đặt cách đều red_nearest và blue_nearest.
        """
        HW, HH = self.HW, self.HH
        n = self.n_per_team
        margin = PLYR_R + 4.0   # khoảng cách tối thiểu với tường

        # RED: nửa trái x ∈ [-HW+margin, -margin]
        self.red_players  = []
        for _ in range(n):
            pos = self._rand_in_rect(-HW + margin, -margin, -HH + margin, HH - margin,
                                     existing=self.red_players)
            self.red_players.append(
                Disc(pos[0], pos[1], 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            )

        # BLUE: nửa phải x ∈ [margin, HW-margin]
        self.blue_players = []
        for _ in range(n):
            pos = self._rand_in_rect(margin, HW - margin, -HH + margin, HH - margin,
                                     existing=self.blue_players)
            self.blue_players.append(
                Disc(pos[0], pos[1], 0, 0, PLYR_R, PLYR_IMASS, PLYR_BCOEF, PLYR_DAMP)
            )

        # Bóng: midpoint của red_nearest và blue_nearest (equidistant)
        bx, by = self._equidistant_ball_pos()
        self.ball = Disc(bx, by, 0.0, 0.0, BALL_R, BALL_IMASS, BALL_BCOEF, BALL_DAMP)

    def _rand_in_rect(
        self,
        x_lo: float, x_hi: float,
        y_lo: float, y_hi: float,
        existing: list[Disc],
        max_tries: int = 100,
    ) -> Tuple[float, float]:
        """Random (x,y) trong hình chữ nhật, không chồng lên discs đã có."""
        min_sep = PLYR_R * 2 + 6.0
        for _ in range(max_tries):
            x = float(self._rng.uniform(x_lo, x_hi))
            y = float(self._rng.uniform(y_lo, y_hi))
            if all(math.hypot(x - d.x, y - d.y) >= min_sep for d in existing):
                return (x, y)
        # Fallback: trả về trung tâm vùng
        return ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)

    def _equidistant_ball_pos(self) -> Tuple[float, float]:
        """
        Tìm điểm bóng cách đều red_nearest và blue_nearest.
        → Midpoint của (red cầu thủ gần nhất về phía giữa) và (blue cầu thủ gần nhất về phía giữa).
        """
        # Cầu thủ RED gần trục Y nhất (x lớn nhất trong nửa trái)
        red_nearest  = max(self.red_players,  key=lambda d: d.x)
        # Cầu thủ BLUE gần trục Y nhất (x nhỏ nhất trong nửa phải)
        blue_nearest = min(self.blue_players, key=lambda d: d.x)

        bx = (red_nearest.x + blue_nearest.x) / 2.0
        by = (red_nearest.y + blue_nearest.y) / 2.0
        # Clamp để bóng không vượt ra ngoài sân
        HW, HH = self.HW, self.HH
        bx = float(np.clip(bx, -HW + BALL_R + 1, HW - BALL_R - 1))
        by = float(np.clip(by, -HH + BALL_R + 1, HH - BALL_R - 1))
        return bx, by

    def _rebuild_poles(self) -> None:
        """Tái tạo 4 cột gôn tĩnh (goal poles) theo HW và goal_y hiện tại."""
        HW, gy = self.HW, self.goal_y
        self._poles = [
            Disc( HW, -gy, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),  # Right Top
            Disc( HW,  gy, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),  # Right Bottom
            Disc(-HW, -gy, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),  # Left Top
            Disc(-HW,  gy, 0, 0, POLE_R, POLE_IMASS, POLE_BCOEF, 1.0),  # Left Bottom
        ]

    #  Physics tick 
    def _tick(self, actions: list) -> int:
        """
        Một physics tick.
        Args:
            actions: list of [dir_idx, kick] for each agent
        Returns: 0=bình thường, 1=RED ghi bàn (Blue goal bên phải), -1=BLUE ghi bàn.
        """
        ball   = self.ball
        n      = self.n_per_team
        HW, HH = self.HW, self.HH
        gy     = self.goal_y

        all_players = self.red_players + self.blue_players
        
        #  1. Kick & Acceleration 
        for i, ag in enumerate(all_players):
            dir_idx = int(actions[i][0])
            kick    = int(actions[i][1])
            dx, dy  = DIR_MAP[dir_idx]

            # Logic ready to shot
            # states: 0: idle, 1: ready_to_shot, 2: already_shot
            if kick == 0:
                self.ready_to_shot_states[i] = 0
                actually_kick = False
                ready_to_shot = False
            else:
                if self.ready_to_shot_states[i] == 0:
                    self.ready_to_shot_states[i] = 1
                
                if self.ready_to_shot_states[i] == 1:
                    actually_kick = True
                    ready_to_shot = True
                else: # State 2: already_shot
                    actually_kick = False
                    ready_to_shot = False

            # Kick: áp impulse lên bóng nếu trong tầm
            if actually_kick:
                dx_b  = ball.x - ag.x; dy_b = ball.y - ag.y
                dist  = math.hypot(dx_b, dy_b)
                if dist > 0 and dist - ag.radius - ball.radius < KICK_RANGE:
                    nx, ny = dx_b / dist, dy_b / dist
                    ball.xs += nx * KICK_STR
                    ball.ys += ny * KICK_STR
                    # Nếu sút trúng bóng thì chuyển sang trạng thái đã sút (ngắt ready to shot)
                    self.ready_to_shot_states[i] = 2

            # Acceleration
            ln  = math.hypot(dx, dy)
            # Chỉ đi chậm (bằng tốc độ sút) nếu đang ở chế độ ready_to_shot (trạng thái 1)
            # Hoặc vừa mới sút xong nhưng vẫn giữ phím sút thì tốc độ trở lại bình thường
            acc = PLYR_KICK_ACC if ready_to_shot else PLYR_ACC
            
            if ln > 0:
                ndx, ndy = dx / ln, dy / ln
                ag.xs += ndx * acc
                ag.ys += ndy * acc

        #  2. Move 
        ball.x += ball.xs; ball.y += ball.ys
        for ag in all_players:
            ag.x += ag.xs; ag.y += ag.ys

        #  3. Disc-Disc collisions 
        all_discs = [ball] + all_players
        nd = len(all_discs)
        for i in range(nd):
            for j in range(i + 1, nd):
                _resolve_dd(all_discs[i], all_discs[j])

        #  4. Goal pole collisions 
        prev_ball_x = ball.x - ball.xs  # vị trí bóng trước tick (approx)
        for pole in self._poles:
            _resolve_dd(ball, pole)
            for ag in all_players:
                _resolve_dd(ag, pole)

        #  5. Wall collisions — Ball 
        # Top / Bottom (bóng nảy)
        if ball.y - ball.radius < -HH:
            ball.y  = -HH + ball.radius
            ball.ys = -ball.ys * ball.bcoef
        if ball.y + ball.radius > HH:
            ball.y  =  HH - ball.radius
            ball.ys = -ball.ys * ball.bcoef

        # Left wall (Red goal at x = -HW)
        if ball.x - ball.radius < -HW:
            # Nội suy y tại x = -HW để kiểm tra gôn (chuẩn HaxBall)
            prev_x = ball.x - ball.xs
            if prev_x != ball.x:
                t      = (-HW - prev_x) / (ball.x - prev_x)
                cross_y = (ball.y - ball.ys) + t * ball.ys
            else:
                cross_y = ball.y
            if abs(cross_y) <= gy:
                return -1   # BLUE ghi bàn (bóng vào gôn Red)
            # Nảy khỏi tường trái
            ball.x  = -HW + ball.radius
            ball.xs = -ball.xs * ball.bcoef

        # Right wall (Blue goal at x = +HW)
        elif ball.x + ball.radius > HW:
            prev_x = ball.x - ball.xs
            if prev_x != ball.x:
                t       = (HW - prev_x) / (ball.x - prev_x)
                cross_y = (ball.y - ball.ys) + t * ball.ys
            else:
                cross_y = ball.y
            if abs(cross_y) <= gy:
                return 1    # RED ghi bàn (bóng vào gôn Blue)
            # Nảy khỏi tường phải
            ball.x  =  HW - ball.radius
            ball.xs = -ball.xs * ball.bcoef

        #  6. Wall collisions — Players (outer boundary) 
        oxW = HW + OUTER_PAD; oxH = HH + OUTER_PAD
        for ag in all_players:
            if ag.x - ag.radius < -oxW: ag.x  = -oxW + ag.radius; ag.xs =  abs(ag.xs) * 0.3
            if ag.x + ag.radius >  oxW: ag.x  =  oxW - ag.radius; ag.xs = -abs(ag.xs) * 0.3
            if ag.y - ag.radius < -oxH: ag.y  = -oxH + ag.radius; ag.ys =  abs(ag.ys) * 0.3
            if ag.y + ag.radius >  oxH: ag.y  =  oxH - ag.radius; ag.ys = -abs(ag.ys) * 0.3

        #  7. Damping 
        ball.xs *= ball.damp; ball.ys *= ball.damp
        for ag in all_players:
            ag.xs *= ag.damp; ag.ys *= ag.damp

        return 0

    #  Observation builder 
    def _build_all_obs(self) -> list[np.ndarray]:
        """Xây dựng obs cho tất cả n_agents (RED trước, BLUE sau)."""
        obs_list = []
        # RED agents (team_id=1, flip=+1)
        for agent_idx, ag in enumerate(self.red_players):
            obs_list.append(self._get_obs_for(ag, team_id=1, agent_idx=agent_idx))
        # BLUE agents (team_id=2, flip=-1)
        for agent_idx, ag in enumerate(self.blue_players):
            obs_list.append(self._get_obs_for(ag, team_id=2, agent_idx=self.n_per_team + agent_idx))
        return obs_list

    def _get_obs_for(self, agent: Disc, team_id: int, agent_idx: int = 0) -> np.ndarray:
        """
        Build observation vector cho một agent.

        team_id=1 → RED (goal tấn công ở x=+HW, flip=+1)
        team_id=2 → BLUE (goal tấn công ở x=-HW, flip=-1 → x bị lật để policy dùng chung)

        Layout:
          [0..3]   Field constants : goal_y/N, HH/N, HW/N, team_flag
          [4..7]   Agent↔Ball      : d_bx/N, d_by/N, surf_dist/DIAG, kick_margin
          [8..18]  Dynamic state   : bx,by,bxs,bys, mx,my,mxs,mys, speed, rvx,rvy
          [19..22] Game state      : time_remaining, possession, ready_to_shot, time_speed
          [23..62] Teammates×4     : (x,y,xs,ys,d_me_x,d_me_y,d_ball_x,d_ball_y,dist_ball,kick_margin) ×4
          [63..112] Opponents×5    : same layout ×5
        """
        obs  = np.zeros(OBS_DIM, dtype=np.float32)
        ball = self.ball
        flip = 1.0 if team_id == 1 else -1.0

        bx,  by  = ball.x,  ball.y
        bxs, bys = ball.xs, ball.ys
        mx,  my  = agent.x, agent.y
        mxs, mys = agent.xs, agent.ys

        surf_dist = max(0.0, math.hypot(mx - bx, my - by) - PLYR_R - BALL_R)

        i = 0

        # Section 1 — Field constants (4)
        obs[i] = self.goal_y / NORM;                    i += 1
        obs[i] = self.HH / NORM;                        i += 1
        obs[i] = self.HW / NORM;                        i += 1
        obs[i] = 0.0 if team_id == 1 else 1.0;         i += 1  # 0=RED, 1=BLUE

        # Section 2 — Agent ↔ Ball (4)
        obs[i] = flip * (bx - mx) / NORM;  i += 1
        obs[i] = (by - my) / NORM;         i += 1
        obs[i] = surf_dist / DIAG;         i += 1
        obs[i] = (surf_dist - KICK_RANGE) / DIAG; i += 1

        # Section 3 — Dynamic state (11)
        obs[i] = flip * bx  / NORM;         i += 1
        obs[i] = by  / NORM;                i += 1
        obs[i] = flip * bxs / MAX_SPEED;    i += 1
        obs[i] = bys / MAX_SPEED;           i += 1
        obs[i] = flip * mx  / NORM;         i += 1
        obs[i] = my  / NORM;                i += 1
        obs[i] = flip * mxs / MAX_SPEED;    i += 1
        obs[i] = mys / MAX_SPEED;           i += 1
        obs[i] = math.hypot(mxs, mys) / MAX_SPEED;     i += 1
        obs[i] = flip * (mxs - bxs) / MAX_SPEED;       i += 1
        obs[i] = (mys - bys) / MAX_SPEED;               i += 1

        # Section 4 — Game state (7) including ready_to_shot, time_speed, and scores
        obs[i] = max(0.0, 1.0 - self.step_count / max(1, self.max_steps)); i += 1
        obs[i] = 0.0;                       i += 1  # possession (TODO)
        # ready_to_shot flag (1.0 if in ready to shot mode, else 0.0)
        obs[i] = 1.0 if self.ready_to_shot_states[agent_idx] == 1 else 0.0; i += 1
        # Tổng thời gian trận (phút) / 10
        obs[i] = (self.ep_seconds / 60.0) / 10.0; i += 1
        # Score features (normalized by goal_limit)
        gl = float(self.goal_limit)
        obs[i] = 1.0 / gl;                                                   i += 1  # goal_step
        my_score  = float(self.red_score  if team_id == 1 else self.blue_score)
        opp_score = float(self.blue_score if team_id == 1 else self.red_score)
        obs[i] = my_score  / gl;                                             i += 1  # my_goals / goal_limit
        obs[i] = opp_score / gl;                                             i += 1  # opp_goals / goal_limit

        # Section 5 — Teammates ×4 (9 features each)
        teammates = [p for p in (self.red_players if team_id == 1 else self.blue_players)
                     if p is not agent]
        for slot in range(N_TM_SLOTS):
            if slot < len(teammates):
                t = teammates[slot]
                obs[i]   = flip * t.x / NORM;                         i += 1
                obs[i]   = t.y / NORM;                                 i += 1
                obs[i]   = flip * t.xs / MAX_SPEED;                    i += 1
                obs[i]   = t.ys / MAX_SPEED;                           i += 1
                obs[i]   = flip * (t.x - mx) / NORM;                   i += 1
                obs[i]   = (t.y - my) / NORM;                          i += 1
                obs[i]   = flip * (bx - t.x) / NORM;                   i += 1
                obs[i]   = (by - t.y) / NORM;                          i += 1
                t_surf   = max(0.0, math.hypot(t.x-bx, t.y-by) - PLYR_R - BALL_R)
                obs[i]   = t_surf / DIAG;                               i += 1
                obs[i]   = (t_surf - KICK_RANGE) / DIAG;                i += 1
            else:
                i += 10  # zero-padded

        # Section 6 — Opponents ×5 (9 features each)
        opponents = self.blue_players if team_id == 1 else self.red_players
        for slot in range(N_OPP_SLOTS):
            if slot < len(opponents):
                o = opponents[slot]
                obs[i]   = flip * o.x / NORM;                          i += 1
                obs[i]   = o.y / NORM;                                  i += 1
                obs[i]   = flip * o.xs / MAX_SPEED;                     i += 1
                obs[i]   = o.ys / MAX_SPEED;                            i += 1
                obs[i]   = flip * (o.x - mx) / NORM;                    i += 1
                obs[i]   = (o.y - my) / NORM;                           i += 1
                obs[i]   = flip * (bx - o.x) / NORM;                    i += 1
                obs[i]   = (by - o.y) / NORM;                           i += 1
                o_surf   = max(0.0, math.hypot(o.x-bx, o.y-by) - PLYR_R - BALL_R)
                obs[i]   = o_surf / DIAG;                                i += 1
                obs[i]   = (o_surf - KICK_RANGE) / DIAG;                 i += 1
            else:
                i += 10  # zero-padded

        assert i == OBS_DIM, f"Obs pointer mismatch: got {i}, expected {OBS_DIM}"
        return obs

    def render(self):
        pass

    def close(self):
        pass


 
# Quick sanity test
 
if __name__ == '__main__':
    print("=== HaxballEnv Quick Sanity Test ===\n")

    configs = [
        dict(n_per_team=1, HW=368, HH=171, goal_y=64,  spawn_mode='haxball', label='1v1 haxball-spawn'),
        dict(n_per_team=2, HW=520, HH=242, goal_y=76,  spawn_mode='haxball', label='2v2 haxball-spawn'),
        dict(n_per_team=3, HW=700, HH=320, goal_y=85,  spawn_mode='random',  label='3v3 random-spawn'),
        dict(n_per_team=4, HW=700, HH=320, goal_y=100, spawn_mode='random',  label='4v4 random-spawn'),
    ]

    for cfg in configs:
        label = cfg.pop('label')
        env   = HaxballEnv(**cfg, time_limit_min=3, seed=42)
        obs, _ = env.reset()

        print(f"[{label}]")
        print(f"  n_agents={env.n_agents}, OBS_DIM={OBS_DIM}")
        print(f"  Field: HW={env.HW}, HH={env.HH}, goal_y={env.goal_y}")
        print(f"  Ball spawn  : ({env.ball.x:.1f}, {env.ball.y:.1f})")
        for j, p in enumerate(env.red_players):
            print(f"  RED[{j}] spawn: ({p.x:.1f}, {p.y:.1f})")
        for j, p in enumerate(env.blue_players):
            print(f"  BLU[{j}] spawn: ({p.x:.1f}, {p.y:.1f})")
        print(f"  obs[0] shape={obs[0].shape}, sample={obs[0][:6]}")

        # Run 10 random steps
        for _ in range(10):
            acts = [env.single_action_space.sample() for _ in range(env.n_agents)]
            obs, rews, term, trunc, info = env.step(acts)
            if term or trunc:
                obs, _ = env.reset()

        print(f"  10 steps OK  rewards sample={rews}\n")

    print("All tests passed!")
