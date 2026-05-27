"""
2D Spatial Episode Trajectory Visualization

Generates generalized spatial visualization of episode trajectories showing:
- Human agent path through task steps (circular layout)
- Assistant interventions (red stars)
- Failure events (red X)
- Memory state evolution (heatmap)

Works for any task (8-20 steps), not kitchen-specific.

Usage:
    python visualize_episode_trajectory.py --task make_cereal --regime balanced
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, Tuple, List, Any

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import SimulationParams
from task_definitions import get_task_definition, TaskDefinition, create_per_step_failure_costs
from train_rl_policy import GymWrapperEnv
from stable_baselines3 import PPO


class GeneralizedTaskLayout:
    """Generate 2D spatial layout for arbitrary N-step task."""

    def __init__(self, task_def: TaskDefinition):
        self.task_def = task_def
        self.n_steps = len(task_def.steps)
        self.step_positions = self._compute_circular_layout()
        self.step_colors = self._assign_colors_by_criticality()

    def _compute_circular_layout(self) -> Dict[int, Tuple[float, float]]:
        """Arrange steps in circle with radius 5."""
        radius = 5.0
        positions = {}
        for i in range(self.n_steps):
            angle = (2 * np.pi / self.n_steps) * i - np.pi / 2  # Start at top
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x, y)
        return positions

    def _assign_colors_by_criticality(self) -> Dict[int, str]:
        """Map criticality values to colors."""
        colors = {}
        for i, step in enumerate(self.task_def.steps):
            if step.criticality == 0.0:
                colors[i] = '#D3D3D3'  # Gray (trivial)
            elif step.criticality < 1.5:
                colors[i] = '#FF8C00'  # Orange (moderate)
            else:
                colors[i] = '#FF0000'  # Red (critical)
        return colors

    def draw_layout(self, ax: plt.Axes):
        """Draw static task structure (step circles with labels)."""
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw steps as circles
        for i, (x, y) in self.step_positions.items():
            circle = Circle((x, y), 0.4, facecolor=self.step_colors[i],
                          edgecolor='black', linewidth=2, zorder=5)
            ax.add_patch(circle)

            # Step number inside circle
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=10, weight='bold', zorder=6)

            # Step name label outside
            label_x = x * 1.25
            label_y = y * 1.25
            step_name = self.task_def.steps[i].name[:10]  # Truncate if long
            ax.text(label_x, label_y, step_name, ha='center',
                   fontsize=7, style='italic', zorder=4)


class TrajectoryVisualizer:
    """Visualize episode trajectory on generalized task layout."""

    def __init__(self, history: Dict, params: SimulationParams, task_def: TaskDefinition):
        self.history = history
        self.params = params
        self.task_def = task_def
        self.layout = GeneralizedTaskLayout(task_def)

    def plot_static_trajectory(self, output_path: Path):
        """Create 2-panel static trajectory visualization."""
        fig, (ax_task, ax_memory) = plt.subplots(1, 2, figsize=(16, 8))

        # Left panel: Task space with trajectory
        self.layout.draw_layout(ax_task)
        self._draw_trajectory(ax_task)
        self._draw_interventions(ax_task)
        self._draw_failures(ax_task)
        ax_task.set_title('Episode Trajectory', fontsize=14, weight='bold')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, alpha=0.3, label='Agent Path'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                  markersize=15, label='Intervention'),
            Line2D([0], [0], marker='x', color='red', markersize=10,
                  linewidth=2, label='Failure')
        ]
        ax_task.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Right panel: Memory heatmap
        self._plot_memory_heatmap(ax_memory)

        plt.tight_layout()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to: {output_path}")
        plt.close()

    def _draw_trajectory(self, ax: plt.Axes):
        """Draw agent path through task space."""
        if len(self.history['step_progression']) == 0:
            return

        # Extract positions
        positions = []
        for step_idx, tau in self.history['step_progression']:
            # Skip invalid step indices (e.g., terminal state)
            if step_idx >= self.layout.n_steps:
                continue
            x, y = self.layout.step_positions[step_idx]
            positions.append((x, y))

        if len(positions) < 2:
            return

        # Draw path line
        xs, ys = zip(*positions)
        ax.plot(xs, ys, 'b-', alpha=0.3, linewidth=2, zorder=2)

        # Draw position markers colored by time progression
        subsample_rate = max(1, len(positions) // 100)  # Subsample for large episodes
        for i in range(0, len(positions), subsample_rate):
            x, y = positions[i]
            # Color by time: darker = later in episode
            color = plt.cm.viridis(i / len(positions))
            ax.scatter(x, y, c=[color], s=20, zorder=3, alpha=0.7)

    def _draw_interventions(self, ax: plt.Axes):
        """Mark intervention locations with red stars."""
        intervention_positions = []

        for tick, action in enumerate(self.history['actions_assistant']):
            if action != 0:  # Not silent
                if tick < len(self.history['step_progression']):
                    step_idx, _ = self.history['step_progression'][tick]
                    # Skip invalid step indices
                    if step_idx >= self.layout.n_steps:
                        continue
                    x, y = self.layout.step_positions[step_idx]
                    intervention_positions.append((x, y))

        if intervention_positions:
            xs, ys = zip(*intervention_positions)
            ax.scatter(xs, ys, marker='*', c='red', s=300,
                      edgecolor='white', linewidth=1.5, zorder=10)

    def _draw_failures(self, ax: plt.Axes):
        """Mark failure locations with red X."""
        failure_positions = []

        for tick, failed in enumerate(self.history['failures']):
            if failed and tick < len(self.history['step_progression']):
                step_idx, _ = self.history['step_progression'][tick]
                # Skip invalid step indices
                if step_idx >= self.layout.n_steps:
                    continue
                x, y = self.layout.step_positions[step_idx]
                failure_positions.append((x, y))

        for x, y in failure_positions:
            # Draw X
            size = 0.3
            ax.plot([x-size, x+size], [y-size, y+size], 'r-',
                   linewidth=3, zorder=11)
            ax.plot([x-size, x+size], [y+size, y-size], 'r-',
                   linewidth=3, zorder=11)

    def _plot_memory_heatmap(self, ax: plt.Axes):
        """Plot memory levels over time as heatmap."""
        if len(self.history['observations']) == 0:
            ax.text(0.5, 0.5, 'No Memory Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        # Extract memory matrix: rows = steps, cols = ticks
        memory_matrix = []
        for obs in self.history['observations']:
            memory_matrix.append(obs['memory'])

        memory_matrix = np.array(memory_matrix).T  # Shape: (n_steps, n_ticks)

        # Plot heatmap
        im = ax.imshow(memory_matrix, aspect='auto', cmap='viridis',
                      interpolation='nearest', origin='lower')

        ax.set_xlabel('Time (ticks)', fontsize=12)
        ax.set_ylabel('Step Index', fontsize=12)
        ax.set_title('Memory Evolution', fontsize=14, weight='bold')

        # Set y-ticks to step indices
        ax.set_yticks(range(self.layout.n_steps))
        ax.set_yticklabels(range(self.layout.n_steps))

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Memory Level')
        cbar.set_label('Memory Level', fontsize=10)

    def create_animation(self, output_path: Path, fps: int = 10):
        """Create animated GIF of episode progression.

        Args:
            output_path: Path to save animation (.gif)
            fps: Frames per second
        """
        print(f"Creating animation with {len(self.history['step_progression'])} frames...")

        fig, (ax_task, ax_memory) = plt.subplots(1, 2, figsize=(16, 8))

        # Draw static layout
        self.layout.draw_layout(ax_task)
        ax_task.set_title('Episode Trajectory (Animated)', fontsize=14, weight='bold')

        # Initialize artists
        path_line, = ax_task.plot([], [], 'b-', alpha=0.3, linewidth=2)
        agent_marker, = ax_task.plot([], [], 'bo', markersize=15, zorder=10)
        intervention_scatter = ax_task.scatter([], [], marker='*', c='red', s=300,
                                              edgecolor='white', linewidth=1.5, zorder=10)
        failure_scatter = ax_task.scatter([], [], marker='x', c='red', s=200,
                                         linewidth=3, zorder=11)

        # Memory display (simplified for animation)
        ax_memory.set_xlim(0, self.layout.n_steps)
        ax_memory.set_ylim(0, 2.0)
        ax_memory.set_xlabel('Step Index', fontsize=12)
        ax_memory.set_ylabel('Memory Level', fontsize=12)
        ax_memory.set_title('Current Memory State', fontsize=14, weight='bold')
        memory_bars = ax_memory.bar(range(self.layout.n_steps),
                                    np.zeros(self.layout.n_steps),
                                    color='blue', alpha=0.7)

        # Track data
        path_positions = []
        intervention_positions = []
        failure_positions = []

        def animate(frame):
            if frame >= len(self.history['step_progression']):
                return

            step_idx, tau = self.history['step_progression'][frame]

            # Update path
            if step_idx < self.layout.n_steps:
                x, y = self.layout.step_positions[step_idx]
                path_positions.append((x, y))

                if len(path_positions) > 1:
                    xs, ys = zip(*path_positions)
                    path_line.set_data(xs, ys)

                # Update agent position
                agent_marker.set_data([x], [y])

            # Update interventions
            if frame < len(self.history['actions_assistant']):
                action = self.history['actions_assistant'][frame]
                if action != 0 and step_idx < self.layout.n_steps:
                    x, y = self.layout.step_positions[step_idx]
                    intervention_positions.append((x, y))
                    if intervention_positions:
                        xs, ys = zip(*intervention_positions)
                        intervention_scatter.set_offsets(np.c_[xs, ys])

            # Update failures
            if frame < len(self.history['failures']):
                if self.history['failures'][frame] and step_idx < self.layout.n_steps:
                    x, y = self.layout.step_positions[step_idx]
                    failure_positions.append((x, y))
                    if failure_positions:
                        xs, ys = zip(*failure_positions)
                        failure_scatter.set_offsets(np.c_[xs, ys])

            # Update memory bars
            if frame < len(self.history['observations']):
                memory = self.history['observations'][frame]['memory']
                for i, bar in enumerate(memory_bars):
                    if i < len(memory):
                        bar.set_height(memory[i])

            return path_line, agent_marker, intervention_scatter, failure_scatter, *memory_bars

        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.history['step_progression']),
                           interval=1000//fps, blit=False, repeat=True)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer)
        print(f"Animation saved to: {output_path}")
        plt.close()


def run_episode_and_capture(task_name: str, regime_name: str) -> Tuple[Dict, SimulationParams, TaskDefinition]:
    """Load model, run one episode, return history and config."""
    print(f"\nRunning episode: {task_name} / {regime_name}")

    # Load task definition
    task_def = get_task_definition(task_name)
    print(f"  Task: {task_name} ({len(task_def.steps)} steps)")

    # Define regime parameters
    regime_configs = {
        'very_high_stakes': {'c_fail': 30.0, 'f0_base': 0.6, 'lambda_forget': 0.05},
        'balanced': {'c_fail': 15.0, 'f0_base': 0.6, 'lambda_forget': 0.05},
        'moderate_low': {'c_fail': 10.0, 'f0_base': 0.6, 'lambda_forget': 0.05}
    }

    regime_config = regime_configs[regime_name]

    # Create per-step failure costs
    c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

    # Create simulation params
    params = SimulationParams(
        c_fail_per_step=c_fail_per_step,
        c_remind=1.0,
        c_nar=0.0,
        c_resp=0.0,
        f0_base=regime_config['f0_base'],
        lambda_forget=regime_config['lambda_forget'],
        delta_reminder=0.6,
        k_memory=3.0,
        step_mean_duration=8,
        obs_noise=0.0,
        c_off_timing=0.5
    )

    # Create environment
    env = GymWrapperEnv(params, task_def)

    # Load trained model
    model_path = PROJECT_ROOT / "models" / regime_name / task_name / "final_model" / "final_model.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"  Loading model from: {model_path}")
    model = PPO.load(str(model_path))

    # Run episode
    print("  Running episode...")
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

    # Extract history from underlying environment
    history = env.env.history

    print(f"  Episode complete: {steps} ticks, reward={episode_reward:.1f}")
    print(f"  Interventions: {sum(1 for a in history['actions_assistant'] if a != 0)}")
    print(f"  Failures: {sum(history['failures'])}")

    return history, params, task_def


def main():
    """Entry point: run episode and visualize."""
    parser = argparse.ArgumentParser(description='Visualize episode trajectory')
    parser.add_argument('--task', type=str, default='make_cereal',
                       help='Task name (default: make_cereal)')
    parser.add_argument('--regime', type=str, default='balanced',
                       choices=['very_high_stakes', 'balanced', 'moderate_low'],
                       help='Cost regime (default: balanced)')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animated GIF instead of static PNG')

    args = parser.parse_args()

    print("=" * 80)
    print("EPISODE TRAJECTORY VISUALIZATION")
    print("=" * 80)

    try:
        # Run episode
        history, params, task_def = run_episode_and_capture(args.task, args.regime)

        # Visualize
        print("\nGenerating visualization...")
        viz = TrajectoryVisualizer(history, params, task_def)

        if args.animate:
            output_path = PROJECT_ROOT / "results" / "videos" / f"trajectory_{args.task}_{args.regime}.gif"
            viz.create_animation(output_path, fps=10)
        else:
            output_path = PROJECT_ROOT / "results" / "figures" / f"trajectory_{args.task}_{args.regime}.png"
            viz.plot_static_trajectory(output_path)

        print("\n✓ Complete!")
        print(f"\nOutput: {output_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
