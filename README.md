# HaxBall AI Training - Multi-Agent PPO

**Thí nghiệm huấn luyện multi-agent agents chơi bóng đá bằng thuật toán PPO (Proximal Policy Optimization).**

Hệ thống này gồm các thành phần chính: xử lý observation, mạng Actor-Critic, và PPO training loop.

## Cấu Trúc

```
training/
├── __init__.py           # Package initialization
├── observations.py       # Observation processing & feature extraction
├── algorithm.py          # PPO training algorithm & neural networks
├── train.py              # Main training loop
└── README.md             # Documentation
```

## Các Thành Phần Chính

### 1. **observations.py** - Xử Lý Observations

**Structured ObservationData:**
Framework cấu trúc observation giúp dễ quản lý và mở rộng. Bao gồm:

```
ObservationData:
├── agent: PlayerInfo                    # Thông tin player của mình
├── teammates: List[PlayerInfo]          # Danh sách đồng đội
├── opponents: List[PlayerInfo]          # Danh sách đối thủ
├── goal_info: GoalInfo                  # Thông tin gôn & vectors
├── half_width, half_height: float       # Kích thước sân
└── ball: BallInfo                       # Thông tin bóng
```

**PlayerInfo** gồm:
- `pos_normalized`: Vị trí chuẩn hóa (-1 to 1)
- `pos_relative_ball`: Vị trí tương đối với bóng
- `dist_to_ball`: Khoảng cách tới bóng
- `can_shoot`: Flag sút bóng
- `velocity`: Vận tốc (vx, vy)

**GoalInfo** gồm:
- `pos_lower`, `pos_upper`: Y-position của cột gôn
- 4 vectors từ 4 góc gôn tới bóng (lower-left, upper-left, lower-right, upper-right)

**BallInfo** gồm:
- `pos`: Vị trí (x, y)
- `velocity`: Vận tốc (vx, vy)
- `bounces_left`: Số lần đập còn lại

**Sử Dụng:**
```python
from observations import ObservationProcessor

processor = ObservationProcessor(obs_dim=110, n_per_team=2)
raw_obs = env.get_observation(agent_id)  # shape (110,)
processed = processor.process_obs(raw_obs)  # normalized & featured
```

### 2. **PPO.py** - Thuật Toán PPO

**ActorCriticNetwork**: Mạng MLP gồm:
- Backbone chia sẻ: trích feature từ observation
- Actor head: policy cho 9 hướng di chuyển
- Kick head: policy cho quyết định sút bóng
- Critic head: value function ước lượng giá trị state

**PPOAgent**: Một agent sử dụng PPO
- `select_action()`: chọn hành động (training/eval)
- `record_step()`: lưu transition
- `compute_advantages()`: tính GAE advantages
- `train_step()`: cập nhật mạng với PPO loss
- `update()`: diễn ra training

**MultiAgentTrainer**: Điều phối multi-agent
- Quản lý N agents
- `select_actions()`: chọn hành động cho tất cả
- `update_all()`: cập nhật tất cả agents
- `save/load_checkpoint()`: lưu/tải mô hình

**PPOConfig**: Cài đặt hyperparameter
```python
config = PPOConfig(
    learning_rate=3e-4,     # Tốc độ học
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    epsilon=0.2,            # PPO clip range
    value_coef=0.5,         # Hệ số value loss
    entropy_coef=0.01,      # Hệ số khám phá
    num_epochs=4,           # Epoch/update
    batch_size=32,          # Kích thước batch
)
```

**Ví dụ sử dụng:**
```python
from training.PPO import PPOAgent, PPOConfig

config = PPOConfig(learning_rate=3e-4)
agent = PPOAgent(obs_dim=110, config=config)

# Thu thập experience
action, kick, log_prob = agent.select_action(obs, training=True)
agent.record_step(obs, (action, kick), reward, done)

# Huấn luyện
agent.update()
```

### 3. **train.py** - Training Loop

*File này sẽ được tạo sau, bào gồm:*
- Thu thập experience từ environment
- Cập nhật mô hình
- Lưu checkpoint
- Đánh giá performance

**Ví dụ Training Loop Code:**
```python
from training import MultiAgentTrainer, PPOConfig, ObservationProcessor
import numpy as np

# 1. Khởi tạo trainer
config = PPOConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    num_epochs=3,
    batch_size=128,
)
trainer = MultiAgentTrainer(n_agents=2, config=config)
processor = ObservationProcessor(obs_dim=65, n_per_team=1)

# 2. Collect experience (mỗi episode)
for episode in range(num_episodes):
    obs_array = env.reset()
    trajectory_length = 0
    
    while not done:
        # Xử lý observations
        processed_obs = [processor.process_obs(o) for o in obs_array]
        
        # Select actions từ PPO
        actions = trainer.select_actions(processed_obs)
        
        # Environment step
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Record transitions
        trainer.record_step(processed_obs, actions, rewards, dones)
        
        obs_array = next_obs
        trajectory_length += 1
        done = any(dones) or trajectory_length >= max_timesteps
    
    # 3. Compute advantages and update
    trainer.update_all()
    
    # 4. Save checkpoint (định kỳ)
    if (episode + 1) % checkpoint_freq == 0:
        trainer.save_checkpoint(f"model_ep{episode+1}.pt")
```

## Quy Trình Huấn Luyện

```
1. Khởi tạo Environment HaxBall
   └─> HaxballEnv (1v1, 2v2, 3v3, 4v4)

2. Mỗi episode:
   a. Reset environment
   b. Khởi tạo observation processors
   
   c. Mỗi bước:
      - Xử lý observation của tất cả agents
      - Chọn hành động (PPO policy)
      - Bước simulation
      - Lưu transitions
   
   d. Tính GAE advantages
   e. Cập nhật models (PPO loss)
   
   f. Lưu checkpoint (định kỳ)
   g. Đánh giá performance (định kỳ)

3. Sau training:
   - Lưu final checkpoint
   - In thống kê
```

## Công Thức PPO Loss

**Policy Loss (PPO-Clip):**
```
L_clip = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```
Giữ thay đổi policy không quá lớn.

**Value Loss:**
```
L_v = MSE(V(s), R_t)
```
Huấn luyện value function ước lượng giá trị state.

**Entropy Bonus:**
```
L_entropy = -β * E[entropy(π)]
```
Khuyến khích exploration.

**Total Loss:**
```
L_total = L_clip + c_v * L_v - c_e * L_entropy
```

## Cấu Trúc Observation Chi Tiết

Hệ thống observation được thiết kế structured để dễ quản lý và mở rộng:

### Classes Chính:

**PlayerInfo** - Thông tin một cầu thủ
```python
class PlayerInfo:
    pos_normalized: np.ndarray       # Vị trí chuẩn (-1 to 1)
    pos_relative_ball: np.ndarray    # Vị trí tương đối bóng
    dist_to_ball: float              # Khoảng cách → bóng
    can_shoot: bool                  # Có thể sút?
    velocity: np.ndarray             # Vận tốc (vx, vy)
```

**GoalInfo** - Thông tin gôn
```python
class GoalInfo:
    pos_lower: float                 # Y-position cột dưới
    pos_upper: float                 # Y-position cột trên
    vec_lower_left_to_ball: np.ndarray    # Vector góc dưới trái → bóng
    vec_upper_left_to_ball: np.ndarray    # Vector góc trên trái → bóng
    vec_lower_right_to_ball: np.ndarray   # Vector góc dưới phải → bóng
    vec_upper_right_to_ball: np.ndarray   # Vector góc trên phải → bóng
```

**BallInfo** - Thông tin bóng
```python
class BallInfo:
    pos: np.ndarray                  # Vị trí (x, y)
    velocity: np.ndarray             # Vận tốc (vx, vy)
    bounces_left: int                # Số lần đập còn lại
```

**ObservationData** - Observation hoàn chỉnh
```python
@dataclass
class ObservationData:
    agent: PlayerInfo                # Player của mình
    teammates: List[PlayerInfo]      # Danh sách đồng đội (max 3)
    opponents: List[PlayerInfo]      # Danh sách đối thủ (max 3)
    goal_info: GoalInfo              # Thông tin gôn
    half_width: float                # Nửa chiều rộng sân
    half_height: float               # Nửa chiều cao sân
    ball: BallInfo                   # Thông tin bóng
    
    def to_flat_array() -> np.ndarray:  # Chuyển thành flat cho NN
```

### Kích Thước Observation:

| Thành phần | Dims | Tổng |
|-----------|------|------|
| Agent | 6 | 6 |
| Teammates (3×) | 7 | 21 |
| Opponents (3×) | 7 | 21 |
| Goal | 10 | 10 |
| Field (half_w, half_h) | 2 | 2 |
| Ball | 5 | 5 |
| **Total** | | **65** |

Obs dims tính theo padding 3 teammates + 3 opponents dù có ít đầu.


Dùng running mean/std (Welford's algorithm):
```
obs_norm = (obs - running_mean) / (running_std + ε)
```

Tính trực tuyến mà không cần lưu tất cả dữ liệu.
Giúp training ổn định hơn.

## Observation và Action

**Observation Structure** (Structured):

Hệ thống observation được định cấu trúc rõ ràng:
```
Thông tin Agent:
├── Vị trí chuẩn (normalized) [-1, 1]
├── Vị trí tương đối bóng
├── Khoảng cách bóng
├── Flag sút bóng (can_shoot)
└── Vận tốc (vx, vy)

Danh sách Teammates (max 3):
└── Như agent (5 fields × N teammates)

Danh sách Opponents (max 3):
└── Như agent (5 fields × N opponents)

Thông tin Gôn:
├── Y-position cột gôn dưới/trên
└── 4 vectors (cột gôn dưới trái → bóng, etc.)

Thông tin Sân:
├── Half_width, half_height
└── (Dùng chuẩn hóa vị trí)

Thông tin Bóng:
├── Vị trí (x, y)
├── Vận tốc (vx, vy)
└── Bounces left (số lần đập còn lại)
```

**Flat Array** (cho Neural Network):
```
Total dims ≈ 6 + (3×7) + (3×7) + 10 + 2 + 5 ≈ 67 dims
  - Agent: 6
  - Teammates: 21 (3×7, padded)
  - Opponents: 21 (3×7, padded)
  - Goal: 10 (2 positions + 4×2 vectors)
  - Field: 2
  - Ball: 5
```

**Action Space:**
- [0]: Di chuyển (9 hướng: đứng, P, T, L, D, PL, TL, DL, DP)
- [1]: Sút bóng (2 lựa chọn: không sút, sút)

**Ví dụ sử dụng:**
```python
from training.observations import ObservationProcessor

# Khởi tạo
processor = ObservationProcessor(obs_dim=110, n_per_team=2)

# Cách 1: Xây dựng dan structured observation
obs_data = processor.create_observation_data(
    agent_pos=np.array([10.0, 5.0]),
    agent_vel=np.array([1.0, 0.5]),
    agent_can_shoot=True,
    teammates_pos=[np.array([15.0, 0.0])],
    teammates_vel=[np.array([0.5, 0.2])],
    teammates_can_shoot=[False],
    opponents_pos=[np.array([-10.0, 5.0]), ...],
    opponents_vel=[...],
    opponents_can_shoot=[...],
    ball_pos=np.array([0.0, 0.0]),
    ball_vel=np.array([0.1, 0.2]),
    ball_bounces=5,
    goal_pos_lower=-85.0,
    goal_pos_upper=85.0,
    half_width=368.0,
    half_height=171.0,
)

# Chuyển thành flat array (chuẩn hóa)
flat_obs = processor.process_observation_data(obs_data)
print(f"Shape: {flat_obs.shape}")  # (~67,)

# Cách 2: Raw observation từ env (backward compatible)
raw_obs = env._get_obs_for(agent_id)
processed = processor.process_obs(raw_obs)
```

## Checkpoints (Lưu Model)

Checkpoint được lưu tại `{checkpoint_dir}/episode_{N}/`:
```
episode_1000/
├── agent_0.pt   # RED team agent 0
├── agent_1.pt   # RED team agent 1
├── agent_2.pt   # BLUE team agent 0
└── agent_3.pt   # BLUE team agent 1 (nếu 2v2)
```

Tải lại mô hình:
```python
from training.PPO import MultiAgentTrainer

trainer = MultiAgentTrainer(n_agents=4, obs_dim=110)
trainer.load_checkpoint('./checkpoints/episode_1000')
```

## Tips Huấn Luyện

1. **GPU**: Tự động dùng CUDA nếu có (xem `PPOConfig.device`)
2. **Batch Size**: Tăng → ổn định hơn nhưng tốn memory
3. **Learning Rate**: Giảm nếu bất ổn, tăng nếu học chậm
4. **GAE Lambda**: 0.95 = high variance, 0.99 = low bias
5. **PPO Epsilon**: 0.2 = cân bằng, nhỏ = an toàn, lớn = nhanh

## Xử Lý Vấn Đề

**Training quá chậm:**
- ↑ Tăng batch_size
- ↓ Giảm num_epochs
- ↑ Tăng learning_rate

**Reward dao động:**
- ↓ Giảm learning_rate
- ↑ Tăng GAE lambda
- ↓ Giảm entropy_coef

**Agents không improve:**
- ↑ Tăng entropy_coef
- ✓ Check observation có ý nghĩa
- ↑ Tăng episode length/độ phức tạp

## Dependencies (Thư Viện)

```
numpy              # Tính toán array
torch              # PyTorch - deep learning
gymnasium          # Environment API
tqdm               # Progress bar
```

Cài đặt:
```bash
pip install torch gymnasium numpy tqdm
```

## Mở Rộng Dự Án

### Thay Đổi Reward

Trên `train.py`:
```python
reward = compute_custom_reward(obs, action, next_obs, goal_scored)
```

### Custom Network

```python
class CustomNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(...)  # LSTM thay MLP
        self.fc = nn.Linear(...)\n```

### Shared Policy

```python
class SharedPPOTrainer:
    def __init__(self, obs_dim, n_agents):
        # Tất cả agents chia network (cooperative)
        self.shared_network = ActorCriticNetwork(obs_dim)
```

## Tài Liệu Tham Khảo

- PPO Paper: https://arxiv.org/abs/1707.06347
- GAE Paper: https://arxiv.org/abs/1506.02438
- PyTorch: https://pytorch.org/
- Gymnasium: https://gymnasium.farama.org/

---

**Thí Nghiệm**: Multi-Agent PPO Training for Soccer (HaxBall)
**Tác Giả**: HaxBall AI Project
**Ngày Cập Nhật**: 2026-04-04
