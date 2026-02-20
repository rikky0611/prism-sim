"""
Training Reward Curves Visualization

Generates 7×3 grid showing reward progression during training for all 21 models
(7 tasks × 3 cost regimes).

Usage:
    python visualize_training_curves.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tensorboard.backend.event_processing import event_accumulator

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_training_summary() -> List[Dict[str, Any]]:
    """Load training summary JSON mapping models to tasks/regimes."""
    summary_path = PROJECT_ROOT / "data" / "results" / "cross_task_training_summary.json"

    with open(summary_path, 'r') as f:
        data = json.load(f)

    # Handle both old and new JSON formats
    return data.get('results', data.get('training_log', []))


def extract_reward_curve(tensorboard_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (timesteps, rewards) from TensorBoard event file.

    Args:
        tensorboard_dir: Path to tensorboard log directory (e.g., PPO_1/)

    Returns:
        Tuple of (timesteps, rewards) arrays
    """
    # Find event file
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"Warning: No event files found in {tensorboard_dir}")
        return np.array([]), np.array([])

    event_file = event_files[0]

    try:
        # Load events
        ea = event_accumulator.EventAccumulator(str(tensorboard_dir))
        ea.Reload()

        # Extract rollout/ep_rew_mean scalar
        if 'rollout/ep_rew_mean' not in ea.Tags()['scalars']:
            print(f"Warning: rollout/ep_rew_mean not found in {tensorboard_dir}")
            return np.array([]), np.array([])

        scalar_events = ea.Scalars('rollout/ep_rew_mean')

        timesteps = np.array([event.step for event in scalar_events])
        rewards = np.array([event.value for event in scalar_events])

        return timesteps, rewards

    except Exception as e:
        print(f"Error reading {tensorboard_dir}: {e}")
        return np.array([]), np.array([])


def smooth_curve(rewards: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential moving average smoothing.

    Args:
        rewards: Array of reward values
        alpha: Smoothing factor (lower = more smoothing)

    Returns:
        Smoothed reward array
    """
    if len(rewards) == 0:
        return rewards

    smoothed = np.zeros_like(rewards)
    smoothed[0] = rewards[0]

    for i in range(1, len(rewards)):
        smoothed[i] = alpha * rewards[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


def get_recent_tensorboard_dirs(tensorboard_base: Path, n: int = 21) -> List[Path]:
    """Get the N most recently modified TensorBoard directories.

    Args:
        tensorboard_base: Base directory containing PPO_* subdirectories
        n: Number of most recent directories to return

    Returns:
        List of Path objects for the N most recent directories
    """
    ppo_dirs = sorted(tensorboard_base.glob("PPO_*"),
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return ppo_dirs[:n]


def plot_training_curves(results: List[Dict[str, Any]], output_path: Path):
    """Create 7×3 grid of training curves.

    Args:
        results: List of training results from summary JSON
        output_path: Path to save output figure
    """
    # Task and regime order
    tasks = ['make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
             'make_stencil', 'cooking', 'latte_making']
    regimes = ['extremely_high', 'moderate', 'extremely_low']

    # Regime colors
    regime_colors = {
        'extremely_high': '#FF4444',  # Red
        'moderate': '#4444FF',        # Blue
        'extremely_low': '#44AA44'    # Green
    }

    # Create 7×3 subplot grid
    fig, axes = plt.subplots(7, 3, figsize=(20, 15), sharex=True)
    fig.suptitle('RL Training Curves: 7 Tasks × 3 Cost Regimes', fontsize=16, weight='bold')

    # Map results to grid positions
    result_grid = {}
    for result in results:
        task = result.get('task', result.get('task_name'))
        regime = result.get('regime', result.get('regime_name'))
        result_grid[(task, regime)] = result

    # TensorBoard directory
    tensorboard_base = PROJECT_ROOT / "models" / "tensorboard"

    # Get 21 most recent TensorBoard directories (from latest 200k training)
    recent_tb_dirs = get_recent_tensorboard_dirs(tensorboard_base, n=21)
    # Reverse to get chronological order (oldest first)
    recent_tb_dirs = list(reversed(recent_tb_dirs))

    # First pass: collect all reward data to determine global y-axis range
    all_rewards = []
    ppo_idx = 0
    for task_idx, task in enumerate(tasks):
        for regime_idx, regime in enumerate(regimes):
            result = result_grid.get((task, regime))
            if result is not None and result.get('success', result.get('status') == 'success'):
                if ppo_idx < len(recent_tb_dirs):
                    tb_dir = recent_tb_dirs[ppo_idx]
                    timesteps, rewards = extract_reward_curve(tb_dir)
                    if len(rewards) > 0:
                        all_rewards.extend(rewards)
                ppo_idx += 1

    # Determine global y-axis range with padding
    if len(all_rewards) > 0:
        global_min = np.min(all_rewards)
        global_max = np.max(all_rewards)
        y_padding = (global_max - global_min) * 0.1
        ylim_min = global_min - y_padding
        ylim_max = min(global_max + y_padding, 5)  # Cap at 5 to show 0 line clearly
    else:
        ylim_min, ylim_max = -250, 0

    # Second pass: plot each task-regime combination with consistent y-axis
    ppo_idx = 0
    for task_idx, task in enumerate(tasks):
        for regime_idx, regime in enumerate(regimes):
            ax = axes[task_idx, regime_idx]

            # Get result for this combination
            result = result_grid.get((task, regime))

            if result is None or not result.get('success', result.get('status') == 'success'):
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 200000)
                ax.set_ylim(ylim_min, ylim_max)
            else:
                # Load tensorboard data from recent directories
                if ppo_idx < len(recent_tb_dirs):
                    tb_dir = recent_tb_dirs[ppo_idx]
                else:
                    tb_dir = tensorboard_base / f"PPO_{ppo_idx + 1}"
                timesteps, rewards = extract_reward_curve(tb_dir)

                if len(rewards) > 0:
                    # Plot raw curve (faint)
                    ax.plot(timesteps, rewards, alpha=0.3, linewidth=0.5,
                           color=regime_colors[regime], label='Raw')

                    # Plot smoothed curve (bold)
                    smoothed = smooth_curve(rewards, alpha=0.1)
                    ax.plot(timesteps, smoothed, linewidth=2,
                           color=regime_colors[regime], label='Smoothed')

                    # Add horizontal line at y=0
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'Data Load Failed', ha='center', va='center', fontsize=10)

                ppo_idx += 1  # Increment for next model

            # Labels and styling
            ax.grid(True, alpha=0.3)

            # Y-axis label (task name on leftmost column)
            if regime_idx == 0:
                ax.set_ylabel(task.replace('_', ' ').title(), fontsize=10, weight='bold')

            # X-axis label (only bottom row)
            if task_idx == len(tasks) - 1:
                ax.set_xlabel('Timesteps', fontsize=10)

            # Column title (only top row)
            if task_idx == 0:
                ax.set_title(regime.replace('_', ' ').title(), fontsize=11, weight='bold')

            # Set consistent limits across all subplots
            ax.set_xlim(0, 200000)
            ax.set_ylim(ylim_min, ylim_max)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    plt.close()


def main():
    """Entry point: load data and generate training curves figure."""
    print("=" * 80)
    print("TRAINING CURVES VISUALIZATION")
    print("=" * 80)

    # Load training summary
    print("\nLoading training summary...")
    results = load_training_summary()
    print(f"Found {len(results)} trained models")

    # Generate plot
    print("\nGenerating training curves (7×3 grid)...")
    output_path = PROJECT_ROOT / "results" / "figures" / "training_curves_all_models.png"
    plot_training_curves(results, output_path)

    print("\n✓ Complete!")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
