"""
utils.py - Utility functions cho training 1v1 Elo-based

Chứa helpful functions cho debugging, visualization, và analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Elo Analysis & Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_elo_distribution(agent_stats: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """
    Phân tích phân phối Elo trong pool.
    
    Tham số:
        agent_stats: Dict[agent_id -> {elo, games_played, ...}]
    
    Trả về:
        Dict với thống kê: mean, std, min, max, median
    """
    elos = np.array([stats['elo'] for stats in agent_stats.values()])
    
    return {
        'mean': float(np.mean(elos)),
        'std': float(np.std(elos)),
        'min': float(np.min(elos)),
        'max': float(np.max(elos)),
        'median': float(np.median(elos)),
        'q1': float(np.percentile(elos, 25)),
        'q3': float(np.percentile(elos, 75)),
    }


def analyze_match_distribution(pairs: List[Tuple[int, int]], 
                               agent_stats: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """
    Phân tích phân phối cân bằng của các cặp.
    
    Tham số:
        pairs: List[(agent1_id, agent2_id)]
        agent_stats: Dict[agent_id -> {elo, ...}]
    
    Trả về:
        Dict với thống kê: mean_diff, std_diff, min_diff, max_diff
    """
    elo_diffs = []
    for agent1_id, agent2_id in pairs:
        elo1 = agent_stats[agent1_id]['elo']
        elo2 = agent_stats[agent2_id]['elo']
        diff = abs(elo1 - elo2)
        elo_diffs.append(diff)
    
    elo_diffs = np.array(elo_diffs)
    
    return {
        'mean_elo_diff': float(np.mean(elo_diffs)),
        'std_elo_diff': float(np.std(elo_diffs)),
        'min_elo_diff': float(np.min(elo_diffs)),
        'max_elo_diff': float(np.max(elo_diffs)),
        'num_pairs': len(pairs),
    }


def get_agent_leaderboard(agent_stats: Dict[int, Dict[str, Any]], 
                          top_k: int = 10) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Lấy leaderboard top K agents.
    
    Trả về: List[(agent_id, stats)]
    """
    sorted_agents = sorted(
        agent_stats.items(),
        key=lambda x: x[1]['elo'],
        reverse=True
    )
    return sorted_agents[:top_k]


def calculate_win_rate_by_elo_bracket(agent_stats: Dict[int, Dict[str, Any]], 
                                      bracket_size: int = 100):
    """
    Tính win rate theo từng Elo bracket.
    
    Tham số:
        bracket_size: Kích thước mỗi bracket (VD: 100 → [1600-1700], [1700-1800], ...)
    
    Trả về:
        Dict[bracket_name -> {count, avg_elo, win_rate}]
    """
    brackets = {}
    
    for agent_id, stats in agent_stats.items():
        elo = stats['elo']
        wins = stats['wins']
        games = stats['games_played']
        
        bracket_min = (int(elo // bracket_size)) * bracket_size
        bracket_name = f"{bracket_min}-{bracket_min + bracket_size}"
        
        if bracket_name not in brackets:
            brackets[bracket_name] = {
                'elos': [],
                'total_wins': 0,
                'total_games': 0,
            }
        
        brackets[bracket_name]['elos'].append(elo)
        brackets[bracket_name]['total_wins'] += wins
        brackets[bracket_name]['total_games'] += games
    
    # Tính toán kết quả
    result = {}
    for bracket_name, data in sorted(brackets.items()):
        count = len(data['elos'])
        avg_elo = np.mean(data['elos'])
        win_rate = (data['total_wins'] / data['total_games'] 
                   if data['total_games'] > 0 else 0.0)
        
        result[bracket_name] = {
            'count': count,
            'avg_elo': float(avg_elo),
            'win_rate': float(win_rate),
            'total_games': data['total_games'],
        }
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint từ file."""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def print_checkpoint_summary(checkpoint_path: str) -> None:
    """In tóm tắt một checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"{'='*60}")
    print(f"Cycle: {checkpoint['cycle']}")
    print(f"Total Matches: {checkpoint['total_matches']}")
    print(f"Pool Size: {checkpoint['pool_size']}")
    
    # Elo stats
    agent_stats = checkpoint['agent_stats']
    elo_dist = analyze_elo_distribution(agent_stats)
    
    print(f"\nElo Distribution:")
    print(f"  Mean: {elo_dist['mean']:.0f}")
    print(f"  Std: {elo_dist['std']:.0f}")
    print(f"  Range: {elo_dist['min']:.0f} - {elo_dist['max']:.0f}")
    print(f"  Median: {elo_dist['median']:.0f}")
    
    # Top agents
    leaderboard = get_agent_leaderboard(agent_stats, top_k=10)
    print(f"\nTop 10 Agents:")
    for rank, (agent_id, stats) in enumerate(leaderboard, 1):
        wr = (stats['wins'] / stats['games_played'] 
              if stats['games_played'] > 0 else 0.0)
        print(f"  {rank:2}. Agent {agent_id:3} | "
              f"Elo: {stats['elo']:7.0f} | "
              f"Record: {stats['wins']:3}W-{stats['losses']:3}L | "
              f"WR: {wr:5.1%}")
    
    print(f"{'='*60}\n")


def compare_checkpoints(checkpoint1_path: str, checkpoint2_path: str) -> None:
    """So sánh hai checkpoints."""
    cp1 = load_checkpoint(checkpoint1_path)
    cp2 = load_checkpoint(checkpoint2_path)
    
    print(f"\n{'='*80}")
    print(f"Comparison: {Path(checkpoint1_path).name} vs {Path(checkpoint2_path).name}")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Checkpoint 1':<20} {'Checkpoint 2':<20}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Cycle', 'cycle'),
        ('Total Matches', 'total_matches'),
        ('Pool Size', 'pool_size'),
    ]
    
    for metric_name, key in metrics:
        v1 = cp1.get(key, 'N/A')
        v2 = cp2.get(key, 'N/A')
        print(f"{metric_name:<30} {str(v1):<20} {str(v2):<20}")
    
    # Elo stats
    elo_dist1 = analyze_elo_distribution(cp1['agent_stats'])
    elo_dist2 = analyze_elo_distribution(cp2['agent_stats'])
    
    print(f"\n{'Elo Statistics':<30}")
    print(f"{'-'*80}")
    
    for stat in ['mean', 'std', 'min', 'max']:
        v1 = elo_dist1[stat]
        v2 = elo_dist2[stat]
        print(f"  {stat.capitalize():<25} {v1:>20.0f} {v2:>20.0f}")
    
    print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_agent_data_csv(agent_stats: Dict[int, Dict[str, Any]], 
                          csv_path: str) -> None:
    """Export agent statistics to CSV."""
    import csv
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['agent_id', 'elo', 'games_played', 'wins', 'losses', 'win_rate']
        )
        writer.writeheader()
        
        for agent_id, stats in sorted(agent_stats.items()):
            win_rate = (stats['wins'] / stats['games_played'] 
                       if stats['games_played'] > 0 else 0.0)
            writer.writerow({
                'agent_id': agent_id,
                'elo': f"{stats['elo']:.0f}",
                'games_played': stats['games_played'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': f"{win_rate:.2%}",
            })
    
    logger.info(f"Exported agent data to {csv_path}")


def export_training_history_csv(training_history: List[Dict[str, Any]],
                                csv_path: str) -> None:
    """Export training history to CSV."""
    import csv
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['cycle', 'pool_size', 'total_matches', 'mean_elo', 'elo_std']
        )
        writer.writeheader()
        
        for cycle_info in training_history:
            agent_stats = cycle_info['agent_stats']
            elo_dist = analyze_elo_distribution(agent_stats)
            
            writer.writerow({
                'cycle': cycle_info['cycle'],
                'pool_size': cycle_info['pool_size'],
                'total_matches': cycle_info['total_matches'],
                'mean_elo': f"{elo_dist['mean']:.0f}",
                'elo_std': f"{elo_dist['std']:.0f}",
            })
    
    logger.info(f"Exported training history to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization (TextBased)
# ═══════════════════════════════════════════════════════════════════════════════

def print_elo_histogram(agent_stats: Dict[int, Dict[str, Any]], 
                        bins: int = 10) -> None:
    """In histogram Elo distribution."""
    elos = np.array([stats['elo'] for stats in agent_stats.values()])
    
    hist, bin_edges = np.histogram(elos, bins=bins)
    
    print(f"\nElo Distribution Histogram:")
    print(f"{'-'*60}")
    
    max_count = max(hist)
    bar_width = 40
    
    for i, count in enumerate(hist):
        bin_min = int(bin_edges[i])
        bin_max = int(bin_edges[i+1])
        bar_len = int((count / max_count) * bar_width)
        bar = '█' * bar_len
        
        print(f"[{bin_min:4d}-{bin_max:4d}] {bar:<{bar_width}} {count:3d}")
    
    print(f"{'-'*60}\n")


def print_agent_table(agent_stats: Dict[int, Dict[str, Any]], 
                      top_k: int = 20) -> None:
    """In bảng agents theo Elo."""
    leaderboard = get_agent_leaderboard(agent_stats, top_k)
    
    print(f"\n{'Rank':<6} {'Agent ID':<12} {'Elo':<10} {'Record':<15} {'Win Rate':<10}")
    print(f"{'-'*60}")
    
    for rank, (agent_id, stats) in enumerate(leaderboard, 1):
        wr = (stats['wins'] / stats['games_played'] 
              if stats['games_played'] > 0 else 0.0)
        record = f"{stats['wins']}-{stats['losses']}"
        
        print(f"{rank:<6} {agent_id:<12} {stats['elo']:<10.0f} {record:<15} {wr:<10.1%}")
    
    print()
