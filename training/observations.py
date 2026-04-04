"""
observations.py - Xử Lý Observation cho Training HaxBall
=========================================================
Thành phần: Chuẩn hóa và trích feature từ raw observation của environment.
Phần của hệ thống huấn luyện multi-agent agents chơi bóng đá bằng PPO.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, NamedTuple
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Normalization Constants
# ─────────────────────────────────────────────────────────────────────────────

# Field dimensions - normalize positions to [-1, 1]
MAX_HALF_WIDTH = 800.0   # Used for both width and height normalization

# Velocity normalization
MAX_SPEED = 15.0

# Time normalization (in seconds, e.g., max 3 minutes = 180 seconds)
MAX_TIME = 360.0

# Ball bounces normalization
MAX_BOUNCES = 4


# ─────────────────────────────────────────────────────────────────────────────
# Observation Structures
# ─────────────────────────────────────────────────────────────────────────────

class PlayerInfo(NamedTuple):
    """Thông tin về một player."""
    pos_normalized: np.ndarray          # Vị trí chuẩn (-1 to 1)
    pos_relative_ball: np.ndarray       # Vị trí tương đối với bóng
    dist_to_ball: float                 # Khoảng cách tới bóng
    can_shoot: bool                     # Flag: có thể sút bóng không
    velocity: np.ndarray                # Tốc độ player (vx, vy)


class BallInfo(NamedTuple):
    """Thông tin về bóng."""
    pos: np.ndarray                     # Vị trí bóng (x, y)
    velocity: np.ndarray                # Tốc độ bóng (vx, vy)
    bounces_left: int                   # Số lần đập còn lại


class GoalInfo(NamedTuple):
    """Thông tin về cột gôn."""
    pos_lower: float                    # Vị trí cột gôn dưới (y)
    pos_upper: float                    # Vị trí cột gôn trên (y)
    vec_lower_left_to_ball: np.ndarray  # Vector từ goal dưới trái → bóng
    vec_upper_left_to_ball: np.ndarray  # Vector từ goal trên trái → bóng
    vec_lower_right_to_ball: np.ndarray # Vector từ goal dưới phải → bóng
    vec_upper_right_to_ball: np.ndarray # Vector từ goal trên phải → bóng
    vec_lower_left_to_agent: np.ndarray # Vector từ goal dưới trái → agent
    vec_upper_left_to_agent: np.ndarray # Vector từ goal trên trái → agent
    vec_lower_right_to_agent: np.ndarray # Vector từ goal dưới phải → agent
    vec_upper_right_to_agent: np.ndarray # Vector từ goal trên phải → agent


@dataclass
class ObservationData:
    """Cấu trúc observation hoàn chỉnh cho một agent."""
    
    # Thông tin agent (player của mình)
    agent: PlayerInfo
    
    # Thông tin đồng đội (teammates)
    teammates: List[PlayerInfo]         # Danh sách đồng đội
    
    # Thông tin đối thủ (opponents)
    opponents: List[PlayerInfo]         # Danh sách đối thủ
    
    # Thông tin gôn
    goal_info: GoalInfo
    
    # Thông tin sân
    half_width: float                   # Nửa chiều rộng sân
    half_height: float                  # Nửa chiều cao sân
    
    # Thông tin bóng
    ball: BallInfo
    
    def to_flat_array(self) -> np.ndarray:
        """Chuyển structured observation thành flat array cho neural network."""
        features = []
        
        # Agent info (values already normalized in create_observation_data)
        features.extend([self.agent.pos_normalized[0], self.agent.pos_normalized[1]])
        features.append(1.0 if self.agent.can_shoot else 0.0)
        features.extend(self.agent.velocity)  # Already normalized by MAX_SPEED
        features.append(self.agent.dist_to_ball)  # Already normalized by MAX_HALF_WIDTH
        features.extend(self.agent.pos_relative_ball)  # Vector từ agent tới ball (normalized)
        
        # Teammates (max 3, already normalized)
        max_teammates = 3
        for i in range(max_teammates):
            if i < len(self.teammates):
                tm = self.teammates[i]
                features.extend(tm.pos_normalized)
                features.append(1.0 if tm.can_shoot else 0.0)
                features.extend(tm.velocity)  # Already normalized by MAX_SPEED
                features.append(tm.dist_to_ball)  # Already normalized by MAX_HALF_WIDTH
                features.extend(tm.pos_relative_ball)  # Vector từ teammate tới ball (normalized)
            else:
                features.extend([0.0] * 9)  # 2 + 1 + 2 + 2 + 2
        
        # Opponents (max 3, already normalized)
        max_opponents = 3
        for i in range(max_opponents):
            if i < len(self.opponents):
                opp = self.opponents[i]
                features.extend(opp.pos_normalized)
                features.append(1.0 if opp.can_shoot else 0.0)
                features.extend(opp.velocity)  # Already normalized by MAX_SPEED
                features.append(opp.dist_to_ball)  # Already normalized by MAX_HALF_WIDTH
                features.extend(opp.pos_relative_ball)  # Vector từ opponent tới ball (normalized)
            else:
                features.extend([0.0] * 9)
        
        # Goal info (already normalized by MAX_HALF_WIDTH in create_observation_data)
        # Goal positions
        features.append(self.goal_info.pos_lower)
        features.append(self.goal_info.pos_upper)
        # Vectors từ 4 cột goal tới ball
        features.extend(self.goal_info.vec_lower_left_to_ball)   # Vector từ goal dưới-trái tới ball
        features.extend(self.goal_info.vec_upper_left_to_ball)   # Vector từ goal trên-trái tới ball
        features.extend(self.goal_info.vec_lower_right_to_ball)  # Vector từ goal dưới-phải tới ball
        features.extend(self.goal_info.vec_upper_right_to_ball)  # Vector từ goal trên-phải tới ball
        # Vectors từ 4 cột goal tới agent chính
        features.extend(self.goal_info.vec_lower_left_to_agent)   # Vector từ goal dưới-trái tới agent
        features.extend(self.goal_info.vec_upper_left_to_agent)   # Vector từ goal trên-trái tới agent
        features.extend(self.goal_info.vec_lower_right_to_agent)  # Vector từ goal dưới-phải tới agent
        features.extend(self.goal_info.vec_upper_right_to_agent)  # Vector từ goal trên-phải tới agent
        
        # Field dimensions (already normalized by MAX_HALF_WIDTH)
        features.append(self.half_width)
        features.append(self.half_height)
        
        # Ball info (already normalized)
        features.extend(self.ball.pos)  # Already normalized by MAX_HALF_WIDTH
        features.extend(self.ball.velocity)  # Already normalized by MAX_SPEED
        # Normalize bounces by MAX_BOUNCES
        features.append(float(self.ball.bounces_left) / MAX_BOUNCES)
        
        return np.array(features, dtype=np.float32)


class ObservationProcessor:
    """
    Lớp để xử lý observation từ HaxballEnv trước khi đưa vào neural network.
    
    Raw observation từ env gồm:
    - Vị trí và velocity của bóng (x, y, vx, vy)
    - Vị trí và velocity của player (x, y, vx, vy)
    - Vị trí các player khác trong tầm nhìn
    - Goal position, scores, thời gian còn lại, ...
    
    Xử lý: Chuẩn hóa + trích engineered features
    Output: Feature vector đã xử lý (float32)
    
    Hoặc có thể trả về ObservationData (structured) và .to_flat_array() cho NN
    """
    
    def __init__(self, obs_dim: int = 110, n_per_team: int = 2, include_history: bool = False):
        """
        Khởi tạo xử lý observation.
        
        Tham số:
            obs_dim: Kích thước raw observation từ env
            n_per_team: Số cầu thủ/team (1v1, 2v2, 3v3, 4v4)
            include_history: Có lưu lịch sử observation (dùng cho LSTM sau)
        """
        self.obs_dim = obs_dim
        self.n_per_team = n_per_team
        self.include_history = include_history
        self.history_len = 4 if include_history else 1
        self.obs_history: List[np.ndarray] = []
        
        # Thống kê chuẩn hóa (sẽ update tức thời hoặc dùng giá trị cố định)
        self.running_mean = np.zeros(obs_dim, dtype=np.float32)
        self.running_std = np.ones(obs_dim, dtype=np.float32)
        self.running_count = 0
    
    def create_observation_data(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        agent_can_shoot: bool,
        teammates_pos: List[np.ndarray],
        teammates_vel: List[np.ndarray],
        teammates_can_shoot: List[bool],
        opponents_pos: List[np.ndarray],
        opponents_vel: List[np.ndarray],
        opponents_can_shoot: List[bool],
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_bounces: int,
        goal_pos_lower: float,
        goal_pos_upper: float,
        half_width: float,
        half_height: float,
    ) -> ObservationData:
        """
        Xây dựng structured ObservationData từ các thành phần riêng lẻ.
        
        Tham số:
            agent_pos, agent_vel: Vị trí & vận tốc player của agent
            agent_can_shoot: Có thể sút bóng hay không
            teammates_pos, teammates_vel, teammates_can_shoot: Danh sách đồng đội
            opponents_pos, opponents_vel, opponents_can_shoot: Danh sách đối thủ
            ball_pos, ball_vel: Vị trí & vận tốc bóng
            ball_bounces: Số lần đập còn lại
            goal_pos_lower, goal_pos_upper: Y-position của cột gôn
            half_width, half_height: Kích thước sân
        
        Ra: ObservationData structured
        """
        # Normalize position (-1 to 1) using MAX_HALF_WIDTH
        agent_pos_norm = np.array([
            agent_pos[0] / MAX_HALF_WIDTH,
            agent_pos[1] / MAX_HALF_WIDTH
        ])
        
        # Normalize agent velocity by MAX_SPEED
        agent_vel_norm = agent_vel / MAX_SPEED
        
        # Agent info
        agent_info = PlayerInfo(
            pos_normalized=agent_pos_norm,
            pos_relative_ball=(agent_pos - ball_pos) / MAX_HALF_WIDTH,
            dist_to_ball=np.linalg.norm(agent_pos - ball_pos) / MAX_HALF_WIDTH,
            can_shoot=agent_can_shoot,
            velocity=agent_vel_norm
        )
        
        # Teammates info
        teammates = []
        for tm_pos, tm_vel, tm_shoot in zip(teammates_pos, teammates_vel, teammates_can_shoot):
            tm_pos_norm = np.array([
                tm_pos[0] / MAX_HALF_WIDTH,
                tm_pos[1] / MAX_HALF_WIDTH
            ])
            tm_vel_norm = tm_vel / MAX_SPEED
            teammates.append(PlayerInfo(
                pos_normalized=tm_pos_norm,
                pos_relative_ball=(tm_pos - ball_pos) / MAX_HALF_WIDTH,
                dist_to_ball=np.linalg.norm(tm_pos - ball_pos) / MAX_HALF_WIDTH,
                can_shoot=tm_shoot,
                velocity=tm_vel_norm
            ))
        
        # Opponents info
        opponents = []
        for opp_pos, opp_vel, opp_shoot in zip(opponents_pos, opponents_vel, opponents_can_shoot):
            opp_pos_norm = np.array([
                opp_pos[0] / MAX_HALF_WIDTH,
                opp_pos[1] / MAX_HALF_WIDTH
            ])
            opp_vel_norm = opp_vel / MAX_SPEED
            opponents.append(PlayerInfo(
                pos_normalized=opp_pos_norm,
                pos_relative_ball=(opp_pos - ball_pos) / MAX_HALF_WIDTH,
                dist_to_ball=np.linalg.norm(opp_pos - ball_pos) / MAX_HALF_WIDTH,
                can_shoot=opp_shoot,
                velocity=opp_vel_norm
            ))
        
        # Goal vectors (từ 4 góc goal tới bóng và tới agent), normalized by MAX_HALF_WIDTH
        goal_left_x = -MAX_HALF_WIDTH
        goal_right_x = MAX_HALF_WIDTH
        
        # Vectors to ball
        vec_lower_left = (ball_pos - np.array([goal_left_x, goal_pos_lower])) / MAX_HALF_WIDTH
        vec_upper_left = (ball_pos - np.array([goal_left_x, goal_pos_upper])) / MAX_HALF_WIDTH
        vec_lower_right = (ball_pos - np.array([goal_right_x, goal_pos_lower])) / MAX_HALF_WIDTH
        vec_upper_right = (ball_pos - np.array([goal_right_x, goal_pos_upper])) / MAX_HALF_WIDTH
        
        # Vectors to agent
        vec_lower_left_agent = (agent_pos - np.array([goal_left_x, goal_pos_lower])) / MAX_HALF_WIDTH
        vec_upper_left_agent = (agent_pos - np.array([goal_left_x, goal_pos_upper])) / MAX_HALF_WIDTH
        vec_lower_right_agent = (agent_pos - np.array([goal_right_x, goal_pos_lower])) / MAX_HALF_WIDTH
        vec_upper_right_agent = (agent_pos - np.array([goal_right_x, goal_pos_upper])) / MAX_HALF_WIDTH
        
        goal_info = GoalInfo(
            pos_lower=goal_pos_lower / MAX_HALF_WIDTH,
            pos_upper=goal_pos_upper / MAX_HALF_WIDTH,
            vec_lower_left_to_ball=vec_lower_left,
            vec_upper_left_to_ball=vec_upper_left,
            vec_lower_right_to_ball=vec_lower_right,
            vec_upper_right_to_ball=vec_upper_right,
            vec_lower_left_to_agent=vec_lower_left_agent,
            vec_upper_left_to_agent=vec_upper_left_agent,
            vec_lower_right_to_agent=vec_lower_right_agent,
            vec_upper_right_to_agent=vec_upper_right_agent,
        )
        
        # Ball info
        ball_info = BallInfo(
            pos=ball_pos / MAX_HALF_WIDTH,
            velocity=ball_vel / MAX_SPEED,
            bounces_left=ball_bounces
        )
        
        return ObservationData(
            agent=agent_info,
            teammates=teammates,
            opponents=opponents,
            goal_info=goal_info,
            half_width=half_width / MAX_HALF_WIDTH,
            half_height=half_height / MAX_HALF_WIDTH,
            ball=ball_info,
        )
    
    @property
    def flat_obs_dim(self) -> int:
        """Tính toán kích thước flat observation vector."""
        # Agent: 6 (pos_x, pos_y, can_shoot, vel_x, vel_y, dist) + 2 (vector_to_ball_x, vector_to_ball_y)
        agent_dim = 8
        
        # Teammates: 3 x 9 (pos_x, pos_y, can_shoot, vel_x, vel_y, dist_to_ball, vec_to_ball_x, vec_to_ball_y)
        teammates_dim = 3 * 9
        
        # Opponents: 3 x 9 (same as teammates)
        opponents_dim = 3 * 9
        
        # Goal: 2 (positions) + 8 (4 goal vectors to ball x 2 dims each) + 8 (4 goal vectors to agent x 2 dims each)
        goal_dim = 2 + 8 + 8
        
        # Field: 2 (half_width, half_height)
        field_dim = 2
        
        # Ball: 4 (pos_x, pos_y, vel_x, vel_y) + 1 (bounces)
        ball_dim = 5
        
        total = agent_dim + teammates_dim + opponents_dim + goal_dim + field_dim + ball_dim
        return total
        
    def process_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Xử lý observation từ environment.
        
        Vào: raw_obs từ env, shape (obs_dim,)
        Ra: feature vector đã xử lý
        """
        # Chuẩn hóa
        normed_obs = self._normalize_obs(raw_obs)
        
        # Trích feature
        features = self._extract_features(normed_obs)
        
        # Lưu lịch sử nếu cần
        if self.include_history:
            self.obs_history.append(features)
            if len(self.obs_history) > self.history_len:
                self.obs_history.pop(0)
            
            # Pad nếu lịch sử chưa đủ
            while len(self.obs_history) < self.history_len:
                self.obs_history.insert(0, features)
            
            stacked = np.stack(self.obs_history, axis=0)  # (history_len, feature_dim)
            return stacked.astype(np.float32)
        
        return features.astype(np.float32)
    
    def process_observation_data(self, obs_data: ObservationData) -> np.ndarray:
        """
        Xử lý structured ObservationData.
        
        Vào: ObservationData
        Ra: Flat feature vector cho neural network
        """
        # Chuyển thành flat array
        flat_array = obs_data.to_flat_array()
        
        # Chuẩn hóa
        normed_array = self._normalize_obs(flat_array)
        
        # Trích feature (hoặc giữ nguyên nếu đã đủ)
        features = self._extract_features(normed_array)
        
        # Lưu lịch sử nếu cần
        if self.include_history:
            self.obs_history.append(features)
            if len(self.obs_history) > self.history_len:
                self.obs_history.pop(0)
            
            while len(self.obs_history) < self.history_len:
                self.obs_history.insert(0, features)
            
            stacked = np.stack(self.obs_history, axis=0)
            return stacked.astype(np.float32)
        
        return features.astype(np.float32)
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Chuẩn hóa observation bằng running mean/std (Welford algorithm)."""
        # Update running stats
        self.running_count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.running_count
        self.running_std = np.sqrt(
            (self.running_std ** 2 + delta * (obs - self.running_mean)) / self.running_count
        )
        
        # Clip std để tránh chia cho số quá nhỏ
        std_clipped = np.maximum(self.running_std, 1e-4)
        return (obs - self.running_mean) / std_clipped
    
    def _extract_features(self, normed_obs: np.ndarray) -> np.ndarray:
        """
        Trích features từ observation đã chuẩn hóa.
        
        Cấu trúc observation (từ haxball_env.py):
        - Bóng: vị trí + vận tốc (4 dims)
        - Player: vị trí + vận tốc (4 dims)
        - Các player khác: vị trí + vận tốc
        - Thông tin gôn, tỷ số, thời gian
        """
        # Lấy normalized observation
        features = normed_obs.copy()
        
        # Thêm engineered features
        # Giả sử 4 dims đầu là vị trí/vận tốc bóng
        if len(normed_obs) >= 8:
            ball_pos = normed_obs[0:2]      # ball x, y
            player_pos = normed_obs[4:6]    # player x, y
            
            # Tính khoảng cách và góc đến bóng
            diff = ball_pos - player_pos
            dist = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            
            # Thêm vào features
            features = np.concatenate([features, [dist, angle]])
        
        return features
    
    def reset_history(self):
        """Xóa lịch sử observation khi episode mới bắt đầu."""
        self.obs_history = []
    
    @property
    def feature_dim(self) -> int:
        """Trả về kích thước feature output."""
        if self.include_history:
            return (self.obs_dim + 2, self.history_len)  # +2 for engineered features
        return self.obs_dim + 2


class MultiAgentObservationBuffer:
    """
    Buffer lưu trữ observation, action, reward của các agents.
    Dùng khi training thu thập data từ environment.
    """
    
    def __init__(self, max_steps: int, n_agents: int, obs_dim: int):
        """
        Khởi tạo buffer.
        
        Tham số:
            max_steps: Max bước/episode
            n_agents: Số agents
            obs_dim: Kích thước feature
        """
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        
        # Reset buffers
        self.reset()
    
    def reset(self):
        """Xóa buffers khi episode mới bắt đầu."""
        self.observations = np.zeros(
            (self.max_steps, self.n_agents, self.obs_dim),
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.max_steps, self.n_agents, 2),  # MultiDiscrete([9, 2])
            dtype=np.int32
        )
        self.rewards = np.zeros(
            (self.max_steps, self.n_agents),
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.max_steps, self.n_agents),
            dtype=bool
        )
        self.next_observations = np.zeros(
            (self.max_steps, self.n_agents, self.obs_dim),
            dtype=np.float32
        )
        self.step_count = 0
    
    def push(self, obs, action, reward, done, next_obs):
        """
        Lưu một step vào buffer.
        
        Vào:
            obs: observation của tất cả agents
            action: action của tất cả agents
            reward: reward của tất cả agents
            done: quá trình kết thúc hay không
            next_obs: observation tiếp theo
        """
        idx = self.step_count
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        
        if isinstance(done, bool):
            self.dones[idx, :] = done
        else:
            self.dones[idx] = done
        
        self.next_observations[idx] = next_obs
        self.step_count += 1
    
    def get_transition(self, index: int) -> Dict[str, np.ndarray]:
        """Lấy một transition tại vị trí index."""
        if index >= self.step_count:
            raise IndexError(f"Index {index} out of range (filled: {self.step_count})")
        
        return {
            'obs': self.observations[index],
            'action': self.actions[index],
            'reward': self.rewards[index],
            'done': self.dones[index],
            'next_obs': self.next_observations[index],
        }
    
    def get_all_transitions(self) -> Dict[str, np.ndarray]:
        """Lấy tất cả transitions đã thu thập."""
        return {
            'obs': self.observations[:self.step_count],
            'action': self.actions[:self.step_count],
            'reward': self.rewards[:self.step_count],
            'done': self.dones[:self.step_count],
            'next_obs': self.next_observations[:self.step_count],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Usage Example
# ─────────────────────────────────────────────────────────────────────────────

"""
Ví dụ sử dụng structured observations:

# Khởi tạo processor
processor = ObservationProcessor(n_per_team=2)

# Lấy data từ environment và xây dựng ObservationData
obs_data = processor.create_observation_data(
    agent_pos=np.array([10.0, 5.0]),
    agent_vel=np.array([1.0, 0.5]),
    agent_can_shoot=True,
    teammates_pos=[np.array([15.0, 0.0])],
    teammates_vel=[np.array([0.5, 0.2])],
    teammates_can_shoot=[False],
    opponents_pos=[np.array([-10.0, 5.0]), np.array([-15.0, -3.0])],
    opponents_vel=[np.array([-1.0, -0.5]), np.array([0.0, 0.0])],
    opponents_can_shoot=[True, False],
    ball_pos=np.array([0.0, 0.0]),
    ball_vel=np.array([0.1, 0.2]),
    ball_bounces=5,
    goal_pos_lower=-85.0,
    goal_pos_upper=85.0,
    half_width=368.0,
    half_height=171.0,
)

# Xử lý thành flat array cho neural network
flat_obs = processor.process_observation_data(obs_data)
print(f"Flat observation shape: {flat_obs.shape}")
print(f"Expected flat obs dim: {processor.flat_obs_dim}")

# Hoặc dùng raw observation từ env
raw_obs = get_obs_from_env()  # Raw array
processed_obs = processor.process_obs(raw_obs)
"""
