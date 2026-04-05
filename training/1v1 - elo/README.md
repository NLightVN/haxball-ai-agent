# 1v1 Elo-Based Curriculum Training

Hệ thống huấn luyện multi-agent với curriculum dựa trên Elo rating cho HaxBall 1v1.

## Các Tính Năng Chính

### 1. **Elo-Based Curriculum Learning**
- Bắt đầu với 10 agents cơ bản
- Thêm 10 agents mới sau mỗi chu kỳ (up to 100 agents)
- Mỗi agent có Elo rating riêng, được update động theo kết quả

### 2. **Bell Curve Opponent Selection**
- Mỗi agent chọn đối thủ dựa trên Gaussian distribution
- Mean của distribution = Elo của agent đó
- Tạo ra các cặp cân bằng cho huấn luyện hiệu quả

### 3. **Parallel Match Training**
- Huấn luyện tất cả cặp song song
- Update Elo + weights cho mỗi agent
- Xây dựng lại cặp cho round tiếp theo

### 4. **Scaling Curriculum**
```
10 agents  × 10 rounds = 50 matches    (5 cặp × 10 trận)
20 agents  × 20 rounds = 200 matches   (10 cặp × 20 trận)
50 agents  × 50 rounds = 1250 matches  (25 cặp × 50 trận)
100 agents × 100 rounds = 5000 matches (50 cặp × 100 trận)
```

### 5. **Adaptive K-Factor**
```python
def k_factor(games_played, pool_size):
    pool_factor = min(1.0, pool_size / 50)
    base_k = max(16, 64 / (1 + games_played / 20))
    return base_k * pool_factor
```

- Pool nhỏ → K thấp (ổn định)
- Pool lớn + ít matches → K cao (thích nghi nhanh)
- Nhiều matches → K thấp (elo ổn định)

## Cấu Trúc File

```
1v1 - elo/
├── train.py           # Main training script
├── config.py          # Hyperparameter configuration
├── utils.py           # Utility functions (coming soon)
├── __init__.py        # Package init
├── README.md          # This file
└── checkpoints/       # Lưu trữ checkpoints
```

## Cách Sử Dụng

### Quick Start - Debug Mode
```bash
cd training/1v1\ -\ elo
python train.py --mode debug
```

### Standard Training
```bash
python train.py --mode standard
```

### Large-Scale Training
```bash
python train.py --mode large
```

## Cấu Hình

### Sửa Default Config
```python
from config import get_config

config = get_config("standard")

# Tùy chỉnh
config.curriculum.initial_pool_size = 20
config.ppo.learning_rate = 1e-4
config.match.num_episodes = 10

# Chạy training
curriculum = EloBasedCurriculum(
    initial_pool_size=config.curriculum.initial_pool_size,
    ppo_config=config.ppo,
)
```

### Custom Configuration
```python
from config import (
    TrainingConfig,
    CurriculumConfig,
    PPOConfig,
    MatchConfig,
    EloConfig,
)

custom_config = TrainingConfig(
    curriculum=CurriculumConfig(
        initial_pool_size=15,
        add_per_cycle=5,
        max_pool_size=50,
    ),
    ppo=PPOConfig(
        learning_rate=2e-4,
        gamma=0.995,
    ),
    match=MatchConfig(
        num_episodes=10,
        time_limit_min=5,
    ),
)
```

## Elo Rating System

### Công Thức Cơ Bản
```
Expected Score: E = 1 / (1 + 10^((opponent_elo - my_elo) / 400))
New Elo: new_elo = old_elo + K * (actual_score - expected_score)
```

### K-Factor Tính Toán
- **Pool Factor**: `min(1.0, pool_size / 50)`
  - Pool nhỏ (< 50) → K nhân với factor < 1
  - Pool lớn (>= 50) → K nhân với 1.0

- **Base K**: `max(16, 64 / (1 + games_played / 20))`
  - Agent mới (games_played = 0) → K = 32
  - Sau 20 trận → K = 32
  - Sau 100 trận → K ≈ 16

## API Overview

### `AgentPool`
Quản lý pool agents với Elo-based selection.

```python
pool = AgentPool(initial_size=10)

# Thêm agents mới
new_ids = pool.add_agents(10)

# Tạo cặp đấu
pairs = pool.create_pairs()  # [(agent1_id, agent2_id), ...]

# Chọn đối thủ theo bell curve
opponent_id = pool.select_opponent(agent_elo=1600, exclude_id=agent_id)

# Lấy thông tin agent
stats = pool.get_agent(agent_id)
print(f"Elo: {stats.elo}, Win rate: {stats.win_rate():.2%}")
```

### `AgentStats`
Thống kê của một agent.

```python
stats = AgentStats(agent_id=0)

# Update Elo
stats.update_elo(opponent_elo=1550, actual_score=1.0, pool_size=50)

# Lấy thông tin
print(f"Elo: {stats.elo}")
print(f"Win rate: {stats.win_rate():.2%}")
```

### `EloBasedCurriculum`
Quản lý toàn bộ curriculum training.

```python
curriculum = EloBasedCurriculum(
    initial_pool_size=10,
    add_per_cycle=10,
    max_pool_size=100,
)

# Chạy training
history = curriculum.train(num_cycles=10)

# Lấy top agents
top_10 = curriculum.get_top_agents(10)

# Lưu checkpoint
curriculum.save_checkpoint("training_checkpoint.json")
```

## Output & Logging

Training sẽ output:
- Real-time logging tới console
- Checkpoint JSON files trong `checkpoints/`
- Top agents by Elo rating

### Example Logs
```
[2024-01-01 12:00:00] [INFO] Starting Elo-based Curriculum Training for 1v1 HaxBall
[2024-01-01 12:00:00] [INFO] Initial pool: 10 agents
[2024-01-01 12:00:05] [INFO] Created 5 pairs from 10 agents
[2024-01-01 12:00:06] [INFO] Match: Agent 0 (1600) vs Agent 5 (1550) → Winner: RED
[2024-01-01 12:00:07] [INFO] Match: Agent 1 (1620) vs Agent 3 (1580) → Winner: BLUE
...
[2024-01-01 12:01:00] [INFO] Cycle 0 completed: 50 matches, pool_size=10
[2024-01-01 12:01:02] [INFO] Added 10 new agents. Pool size: 20
```

## Performance Metrics

Training theo dõi:
- **Win Rate**: Tỷ lệ thắng của mỗi agent
- **Elo Rating**: Đánh giá sức mạnh tương đối
- **Match Distribution**: Phân phối các loại trận đấu
- **Convergence**: Độ ổn định của Elo ratings

## Extension cho HaxballEnv

### Bước 1: Implement `simulate_match()`
```python
def simulate_match(red_agent_id, blue_agent_id, pool, 
                   num_episodes=5, render=False):
    env = HaxballEnv(
        n_per_team=1,  # 1v1
        spawn_mode='random',
        time_limit_min=3,
    )
    
    red_policy = pool.get_agent_ppo(red_agent_id)
    blue_policy = pool.get_agent_ppo(blue_agent_id)
    
    # Chạy matches
    # Update trajectories
    # Return result
```

### Bước 2: Parallel Match Execution
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(simulate_match, red_id, blue_id, pool, num_episodes)
        for red_id, blue_id in pairs
    ]
    
    for future in as_completed(futures):
        result = future.result()
        process_match_result(result, pool)
```

## Debugging Tips

### Log Level Configuration
```python
import logging

logging.basicConfig(level=logging.DEBUG)
# Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Inspecting Agent Pool
```python
# Print all agents
for agent_id, stats in pool.agents.items():
    print(f"Agent {agent_id}: Elo={stats.elo:.0f}, "
          f"Matches={stats.games_played}, WR={stats.win_rate():.2%}")

# Top agents by Elo
sorted_agents = sorted(
    pool.agents.items(),
    key=lambda x: x[1].elo,
    reverse=True
)
```

### Pair Distribution
```python
pairs = pool.create_pairs()
pair_elo_diffs = []
for red_id, blue_id in pairs:
    red_elo = pool.get_agent(red_id).elo
    blue_elo = pool.get_agent(blue_id).elo
    diff = abs(red_elo - blue_elo)
    pair_elo_diffs.append(diff)

print(f"Pair Elo differences: mean={np.mean(pair_elo_diffs):.0f}, "
      f"std={np.std(pair_elo_diffs):.0f}")
```

## Known Limitations

1. **Placeholder Match Simulation**: Hiện tại `simulate_match()` là mô phỏng ngẫu nhiên dựa trên Elo. Cần integrate với HaxballEnv thực tế.

2. **No Parallel Execution Yet**: Training hiện tại chạy tuần tự. Cần thêm ProcessPoolExecutor hoặc AsyncIO.

3. **Fixed Observation Dimension**: Giả định OBS_DIM = 112. Cần sửa nếu observation thay đổi.

4. **Single Action Type**: Hiện tại chỉ support 1v1. Cần mở rộng cho 2v2, 3v3, etc.

## Future Enhancements

- [ ] Integration với HaxballEnv thực tế
- [ ] Parallel match execution với ProcessPoolExecutor
- [ ] Multi-level curriculum (1v1 → 2v2 → 3v3)
- [ ] Adaptive learning rates dựa trên performance
- [ ] Visualization của Elo distributions
- [ ] Integration với TensorBoard
- [ ] Support cho different team sizes

## References

- Elo Rating System: https://en.wikipedia.org/wiki/Elo_rating_system
- PPO Algorithm: https://arxiv.org/abs/1707.06347
- HaxBall Competitive Scene: HaxBall forums

## License

Phần của hệ thống huấn luyện HaxBall AI Agent.
