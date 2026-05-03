"""
example_usage.py - Ví dụ sử dụng training system

Chứa các ví dụ về cách sử dụng Elo-based curriculum training.
"""

import sys
from pathlib import Path

# Thêm paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train import EloBasedCurriculum, AgentPool, simulate_match, process_match_result
from config import TrainingConfig, CurriculumConfig, PPOConfig, MatchConfig, get_config
from utils import (
    print_checkpoint_summary,
    analyze_elo_distribution,
    get_agent_leaderboard,
    print_agent_table,
    print_elo_histogram,
)


def example_1_quick_training():
    """Ví dụ 1: Quick training với cấu hình debug."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Training (Debug Mode)")
    print("="*70)
    
    # Load debug config
    config = get_config("debug")
    
    # Khởi tạo curriculum
    curriculum = EloBasedCurriculum(
        initial_pool_size=config.curriculum.initial_pool_size,
        add_per_cycle=config.curriculum.add_per_cycle,
        max_pool_size=config.curriculum.max_pool_size,
        ppo_config=config.ppo,
    )
    
    print(f"Initial pool size: {len(curriculum.pool.agents)}")
    
    # Train 2 cycles
    curriculum.train(num_cycles=2)
    
    print(f"Final pool size: {len(curriculum.pool.agents)}")
    print(f"Total matches: {curriculum.total_matches}")
    
    # Print leaderboard
    print_agent_table(curriculum.training_history[-1]['agent_stats'], top_k=5)


def example_2_custom_config():
    """Ví dụ 2: Training với custom configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    # Tạo custom config
    custom_config = TrainingConfig(
        curriculum=CurriculumConfig(
            initial_pool_size=8,
            add_per_cycle=4,
            max_pool_size=32,
            initial_num_cycles=3,
        ),
        ppo=PPOConfig(
            learning_rate=2e-4,
            gamma=0.995,
            epsilon=0.15,
        ),
        match=MatchConfig(
            num_episodes=3,
        ),
    )
    
    print(f"Curriculum config:")
    print(f"  Initial pool: {custom_config.curriculum.initial_pool_size}")
    print(f"  Add per cycle: {custom_config.curriculum.add_per_cycle}")
    print(f"  Max pool: {custom_config.curriculum.max_pool_size}")
    
    curriculum = EloBasedCurriculum(
        initial_pool_size=custom_config.curriculum.initial_pool_size,
        add_per_cycle=custom_config.curriculum.add_per_cycle,
        max_pool_size=custom_config.curriculum.max_pool_size,
        ppo_config=custom_config.ppo,
    )
    
    # Train 1 cycle
    curriculum.train(num_cycles=1)
    
    # Phân tích kết quả
    final_history = curriculum.training_history[-1]
    elo_dist = analyze_elo_distribution(final_history['agent_stats'])
    
    print(f"\nElo distribution after 1 cycle:")
    print(f"  Mean: {elo_dist['mean']:.0f}")
    print(f"  Std: {elo_dist['std']:.0f}")
    print(f"  Range: {elo_dist['min']:.0f} - {elo_dist['max']:.0f}")


def example_3_pool_operations():
    """Ví dụ 3: Các operations trên AgentPool."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Agent Pool Operations")
    print("="*70)
    
    # Tạo pool
    pool = AgentPool(initial_size=6)
    
    print(f"Created pool with {len(pool.agents)} agents")
    
    # Print agents
    print("\nInitial agents (sorted by Elo):")
    for agent_id, elo in pool.get_sorted_agents():
        stats = pool.get_agent(agent_id)
        print(f"  Agent {agent_id}: Elo={elo:.0f}")
    
    # Tạo pairs
    pairs = pool.create_pairs()
    print(f"\nCreated {len(pairs)} pairs:")
    for red_id, blue_id in pairs:
        red_elo = pool.get_agent(red_id).elo
        blue_elo = pool.get_agent(blue_id).elo
        diff = abs(red_elo - blue_elo)
        print(f"  Agent {red_id:2} ({red_elo:6.0f}) vs Agent {blue_id:2} ({blue_elo:6.0f}) [diff={diff:5.0f}]")
    
    # Simulate matches and update Elo
    print(f"\nSimulating matches...")
    for red_id, blue_id in pairs:
        result = simulate_match(red_id, blue_id, pool, num_episodes=1)
        process_match_result(result, pool)
    
    # Print updated agents
    print("\nAgents after matches (sorted by Elo):")
    for agent_id, elo in sorted(pool.get_sorted_agents(), key=lambda x: x[1], reverse=True):
        stats = pool.get_agent(agent_id)
        print(f"  Agent {agent_id}: Elo={stats.elo:.0f}, "
              f"Record={stats.wins}W-{stats.losses}L ({stats.win_rate():.1%})")


def example_4_curriculum_progression():
    """Ví dụ 4: Theo dõi curriculum progression."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Curriculum Progression")
    print("="*70)
    
    config = get_config("standard")
    curriculum = EloBasedCurriculum(
        initial_pool_size=config.curriculum.initial_pool_size,
        add_per_cycle=config.curriculum.add_per_cycle,
        max_pool_size=20,  # Reduce để ví dụ nhanh
        ppo_config=config.ppo,
    )
    
    # Train 3 cycles
    curriculum.train(num_cycles=3)
    
    # Print progression
    print(f"{'Cycle':<8} {'Pool Size':<12} {'Matches':<12} {'Mean Elo':<12} {'Elo Std':<12}")
    print("-" * 60)
    
    for cycle_info in curriculum.training_history:
        elo_dist = analyze_elo_distribution(cycle_info['agent_stats'])
        print(f"{cycle_info['cycle']:<8} {cycle_info['pool_size']:<12} "
              f"{cycle_info['total_matches']:<12} {elo_dist['mean']:<12.0f} "
              f"{elo_dist['std']:<12.0f}")


def example_5_checkpoint():
    """Ví dụ 5: Checkpoint save/load."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Checkpoint Operations")
    print("="*70)
    
    config = get_config("debug")
    curriculum = EloBasedCurriculum(
        initial_pool_size=config.curriculum.initial_pool_size,
        add_per_cycle=config.curriculum.add_per_cycle,
        max_pool_size=config.curriculum.max_pool_size,
        ppo_config=config.ppo,
    )
    
    # Train
    curriculum.train(num_cycles=1)
    
    # Save checkpoint
    checkpoint_path = Path("./example_checkpoint.json")
    curriculum.save_checkpoint(str(checkpoint_path))
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Load and print
    if checkpoint_path.exists():
        print_checkpoint_summary(str(checkpoint_path))
        checkpoint_path.unlink()  # Clean up


def example_6_analysis():
    """Ví dụ 6: Analysis utilities."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Analysis Utilities")
    print("="*70)
    
    # Train
    config = get_config("debug")
    curriculum = EloBasedCurriculum(
        initial_pool_size=8,
        add_per_cycle=4,
        max_pool_size=20,
        ppo_config=config.ppo,
    )
    curriculum.train(num_cycles=2)
    
    agent_stats = curriculum.training_history[-1]['agent_stats']
    
    # Analysis
    elo_dist = analyze_elo_distribution(agent_stats)
    leaderboard = get_agent_leaderboard(agent_stats, top_k=5)
    
    print(f"\nElo Distribution:")
    print(f"  Mean: {elo_dist['mean']:.0f}")
    print(f"  Std: {elo_dist['std']:.0f}")
    print(f"  Min: {elo_dist['min']:.0f}")
    print(f"  Max: {elo_dist['max']:.0f}")
    
    print(f"\nTop 5 Agents:")
    for rank, (agent_id, stats) in enumerate(leaderboard, 1):
        wr = stats['wins'] / stats['games_played'] if stats['games_played'] > 0 else 0
        print(f"  {rank}. Agent {agent_id}: {stats['elo']:.0f} ({stats['wins']}W-{stats['losses']}L, {wr:.1%})")
    
    # Histogram
    print_elo_histogram(agent_stats, bins=5)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HAXBALL ELO-BASED CURRICULUM TRAINING - EXAMPLES")
    print("="*70)
    
    try:
        example_1_quick_training()
        example_2_custom_config()
        example_3_pool_operations()
        example_4_curriculum_progression()
        example_5_checkpoint()
        example_6_analysis()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
