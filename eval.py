import sys
import os
import torch
from pathlib import Path

# Thêm project root vào sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from haxball_env import HaxballEnv, OBS_DIM
from training.PPO import PPOAgent

def run_eval(agent_red_path, agent_blue_path, n_matches=5):
    env = HaxballEnv(n_per_team=1, goal_limit=5, ep_seconds=180, spawn_mode='haxball')
    
    agent_red = None
    if agent_red_path and agent_red_path != 'stationary':
        agent_red = PPOAgent(obs_dim=OBS_DIM)
        try:
            agent_red.load_checkpoint(agent_red_path)
        except KeyError:
            # Nếu là snapshot lưu trực tiếp state_dict
            agent_red.network.load_state_dict(torch.load(agent_red_path, map_location='cpu'))
        
    agent_blue = None
    if agent_blue_path and agent_blue_path not in ['stationary', 'none']:
        agent_blue = PPOAgent(obs_dim=OBS_DIM)
        try:
            agent_blue.load_checkpoint(agent_blue_path)
        except KeyError:
            agent_blue.network.load_state_dict(torch.load(agent_blue_path, map_location='cpu'))

    stats = {'RED_WINS': 0, 'BLUE_WINS': 0, 'DRAWS': 0, 'RED_GOALS': 0, 'BLUE_GOALS': 0}

    print(f"\n[EVAL START] RED: {os.path.basename(agent_red_path)} vs BLUE: {os.path.basename(agent_blue_path)}")

    for match_idx in range(n_matches):
        env.reset()
        done = False
        
        # Đưa opponent ra khỏi sân nếu là 'none'
        if agent_blue_path == 'none':
            for ag in env.blue_players:
                ag.x, ag.y = 9999, 9999
                ag.xs, ag.ys = 0, 0
        
        while not done:
            obs_list = env._build_all_obs()
            
            # Red action
            if agent_red:
                move_r, kick_r, _ = agent_red.select_action(obs_list[0], training=False)
            else:
                move_r, kick_r = 0, 0 # stationary
                
            # Blue action
            if agent_blue:
                move_b, kick_b, _ = agent_blue.select_action(obs_list[1], training=False)
            else:
                move_b, kick_b = 0, 0 # stationary
                
            actions = [[move_r, kick_r], [move_b, kick_b]]
            
            # Gọi env.step (đã được vá lỗi tự động đưa bóng về giữa khi có bàn thắng)
            _, _, terminated, truncated, info = env.step(actions)
            
            # Giữ opponent ở ngoài sân
            if agent_blue_path == 'none':
                for ag in env.blue_players:
                    ag.x, ag.y = 9999, 9999
                    ag.xs, ag.ys = 0, 0
                    
            done = terminated or truncated

        red_s = env.red_score
        blue_s = env.blue_score
        stats['RED_GOALS'] += red_s
        stats['BLUE_GOALS'] += blue_s
        
        if red_s > blue_s:
            stats['RED_WINS'] += 1
            winner = "RED WINS"
        elif blue_s > red_s:
            stats['BLUE_WINS'] += 1
            winner = "BLUE WINS"
        else:
            stats['DRAWS'] += 1
            winner = "DRAW"
            
        print(f"  Match {match_idx+1}: {winner} (Score: {red_s} - {blue_s})")

    print(f"--- OVERALL RESULTS ({n_matches} matches) ---")
    print(f"RED  (Bot): {stats['RED_WINS']} Wins, {stats['RED_GOALS']} Goals")
    print(f"BLUE (Bot): {stats['BLUE_WINS']} Wins, {stats['BLUE_GOALS']} Goals")
    print(f"Draws: {stats['DRAWS']}")
    print("---------------------------------------\n")

if __name__ == "__main__":
    cp_dir = Path("training/experiment/1v1/selfplay-snapshot_roundrobin/checkpoints")
    
    snap50 = str(cp_dir / "snapshot_000050.pt")
    snap100 = str(cp_dir / "snapshot_000100.pt")
    
    # 1. Bot 100 vs Không Có Ai (none)
    if os.path.exists(snap100):
        run_eval(snap100, "none", n_matches=3)
    else:
        print(f"File not found: {snap100}")
        
    # 2. Bot 50 vs Bot 100
    if os.path.exists(snap50) and os.path.exists(snap100):
        run_eval(snap50, snap100, n_matches=3)
    else:
        print(f"Checkpoints 50 or 100 not found")
