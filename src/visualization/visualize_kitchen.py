"""
Kitchen Visualization for Procedure Assistant Simulation

Creates animated videos showing:
- Top-down kitchen layout
- Human agent moving through procedural steps
- Assistant notifications (reminders/confirmations)
- Memory state visualization
- Failure events
- Cost accumulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from procedure_assistant_sim import *
import os


# ============================================================================
# KITCHEN LAYOUT DEFINITION
# ============================================================================

class KitchenLayout:
    """Define positions and visuals for kitchen elements"""

    def __init__(self):
        # Kitchen dimensions
        self.width = 10
        self.height = 8

        # Station positions (x, y)
        self.stations = {
            'onion_dispenser': (1, 6),
            'counter': (5, 6),
            'pot': (5, 4),
            'dish_dispenser': (9, 6),
            'serving': (9, 2),
        }

        # Step to station mapping
        self.step_to_station = {
            'get_onion': 'onion_dispenser',
            'deliver_onion': 'pot',
            'wait_cooking': 'pot',
            'get_dish': 'dish_dispenser',
            'serve_soup': 'serving',
        }

        # Colors
        self.colors = {
            'floor': '#F5E6D3',
            'wall': '#8B7355',
            'counter': '#B8860B',
            'onion': '#DAA520',
            'pot': '#CD853F',
            'dish': '#E0E0E0',
            'human': '#4169E1',
            'assistant': '#FF6347',
            'memory': '#32CD32',
            'failure': '#FF0000',
        }

    def draw_kitchen(self, ax):
        """Draw static kitchen elements"""
        ax.clear()

        # Floor
        ax.add_patch(patches.Rectangle(
            (0, 0), self.width, self.height,
            facecolor=self.colors['floor'], edgecolor='none'
        ))

        # Walls
        wall_thickness = 0.3
        ax.add_patch(patches.Rectangle(
            (0, 0), self.width, wall_thickness,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        ax.add_patch(patches.Rectangle(
            (0, self.height - wall_thickness), self.width, wall_thickness,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        ax.add_patch(patches.Rectangle(
            (0, 0), wall_thickness, self.height,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        ax.add_patch(patches.Rectangle(
            (self.width - wall_thickness, 0), wall_thickness, self.height,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))

        # Onion dispenser
        x, y = self.stations['onion_dispenser']
        ax.add_patch(patches.Circle(
            (x, y), 0.4, facecolor=self.colors['onion'],
            edgecolor='black', linewidth=2
        ))
        ax.text(x, y - 0.8, 'ONION', ha='center', fontsize=8, weight='bold')

        # Counter
        x, y = self.stations['counter']
        ax.add_patch(patches.Rectangle(
            (x - 0.5, y - 0.3), 1.0, 0.6,
            facecolor=self.colors['counter'], edgecolor='black', linewidth=2
        ))

        # Pot
        x, y = self.stations['pot']
        ax.add_patch(patches.Circle(
            (x, y), 0.5, facecolor=self.colors['pot'],
            edgecolor='black', linewidth=2
        ))
        ax.text(x, y - 0.9, 'POT', ha='center', fontsize=8, weight='bold')

        # Dish dispenser
        x, y = self.stations['dish_dispenser']
        ax.add_patch(patches.Rectangle(
            (x - 0.3, y - 0.3), 0.6, 0.6,
            facecolor=self.colors['dish'], edgecolor='black', linewidth=2
        ))
        ax.text(x, y - 0.8, 'DISH', ha='center', fontsize=8, weight='bold')

        # Serving area
        x, y = self.stations['serving']
        ax.add_patch(patches.Rectangle(
            (x - 0.5, y - 0.4), 1.0, 0.8,
            facecolor='#FFD700', edgecolor='black', linewidth=2
        ))
        ax.text(x, y, 'SERVE', ha='center', va='center', fontsize=10, weight='bold')

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')


# ============================================================================
# ANIMATION GENERATOR
# ============================================================================

class KitchenAnimator:
    """Generate animated videos of procedure assistant episodes"""

    def __init__(self, history, params, title="Procedure Assistant"):
        self.history = history
        self.params = params
        self.title = title
        self.layout = KitchenLayout()

        # Animation state
        self.frame_rate = 10  # frames per second
        self.ticks_per_frame = 2  # simulation ticks per animation frame

    def get_agent_position(self, tick):
        """Get human agent position at given tick"""
        if tick >= len(self.history['step_progression']):
            step_idx, _ = self.history['step_progression'][-1]
        else:
            step_idx, _ = self.history['step_progression'][tick]

        if step_idx >= len(PROCEDURAL_STEPS):
            return self.layout.stations['serving']

        step_name = PROCEDURAL_STEPS[step_idx]
        station = self.layout.step_to_station[step_name]
        return self.layout.stations[station]

    def animate_frame(self, frame_num):
        """Generate single animation frame"""
        tick = frame_num * self.ticks_per_frame

        if tick >= len(self.history['rewards']):
            tick = len(self.history['rewards']) - 1

        # Clear and redraw kitchen
        self.layout.draw_kitchen(self.ax_kitchen)

        # Get current state
        step_idx, tau = self.history['step_progression'][tick]
        obs = self.history['observations'][tick]
        assistant_action = self.history['actions_assistant'][tick]
        human_action = self.history['actions_human'][tick]
        failure = self.history['failures'][tick]

        # Draw human agent
        agent_pos = self.get_agent_position(tick)
        agent_circle = patches.Circle(
            agent_pos, 0.3,
            facecolor=self.layout.colors['human'],
            edgecolor='white', linewidth=2, zorder=10
        )
        self.ax_kitchen.add_patch(agent_circle)

        # Draw assistant notification if active
        if assistant_action != ASSISTANT_ACTIONS['silent']:
            # Notification bubble
            notify_x, notify_y = agent_pos[0] + 1.2, agent_pos[1] + 0.8

            # Bubble
            bubble = patches.FancyBboxPatch(
                (notify_x - 0.8, notify_y - 0.4), 1.6, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=self.layout.colors['assistant'],
                edgecolor='white', linewidth=2, zorder=15
            )
            self.ax_kitchen.add_patch(bubble)

            # Text
            if assistant_action == ASSISTANT_ACTIONS['confirm']:
                text = "Confirm?"
            else:
                # It's a reminder
                for i in range(len(PROCEDURAL_STEPS)):
                    if assistant_action == ASSISTANT_ACTIONS[f'remind_{i}']:
                        text = f"Remind:\n{PROCEDURAL_STEPS[i][:8]}"
                        break
                else:
                    text = "Remind"

            self.ax_kitchen.text(
                notify_x, notify_y, text,
                ha='center', va='center', fontsize=7,
                color='white', weight='bold', zorder=16
            )

        # Draw failure effect
        if failure:
            # Red X over agent
            x, y = agent_pos
            self.ax_kitchen.plot(
                [x-0.4, x+0.4], [y-0.4, y+0.4],
                'r-', linewidth=4, zorder=12
            )
            self.ax_kitchen.plot(
                [x-0.4, x+0.4], [y+0.4, y-0.4],
                'r-', linewidth=4, zorder=12
            )

        # Draw current step indicator
        if step_idx < len(PROCEDURAL_STEPS):
            step_text = f"Step: {PROCEDURAL_STEPS[step_idx]}"
        else:
            step_text = "Step: DONE"

        self.ax_kitchen.text(
            self.layout.width / 2, 0.5,
            step_text,
            ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Update memory bars
        memory = obs['memory']
        self.ax_memory.clear()

        step_names = [s[:8] for s in PROCEDURAL_STEPS]
        colors = ['green' if i == step_idx else 'gray' for i in range(len(PROCEDURAL_STEPS))]
        bars = self.ax_memory.barh(step_names, memory, color=colors, alpha=0.7, edgecolor='black')

        self.ax_memory.set_xlim(0, 1.0)
        self.ax_memory.set_xlabel('Memory Level', fontsize=9)
        self.ax_memory.set_title('Memory State', fontsize=10, weight='bold')
        self.ax_memory.axvline(0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_memory.grid(axis='x', alpha=0.3)

        # Update cost tracking
        cumulative_reward = sum(self.history['rewards'][:tick+1])
        total_failures = sum(self.history['failures'][:tick+1])
        total_interactions = sum([1 for a in self.history['actions_assistant'][:tick+1]
                                   if a != ASSISTANT_ACTIONS['silent']])

        self.ax_costs.clear()

        # Cost breakdown
        cost_labels = ['Total\nReward', 'Failures', 'Interactions']
        cost_values = [cumulative_reward, -total_failures * self.params.c_fail_base,
                       -total_interactions * self.params.c_int]
        colors = ['blue' if v > 0 else 'red' for v in cost_values]

        bars = self.ax_costs.bar(cost_labels, cost_values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, val in zip(bars, cost_values):
            height = bar.get_height()
            self.ax_costs.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, weight='bold'
            )

        self.ax_costs.axhline(0, color='black', linewidth=1)
        self.ax_costs.set_ylabel('Cost', fontsize=9)
        self.ax_costs.set_title(f'Costs (Tick {tick})', fontsize=10, weight='bold')
        self.ax_costs.grid(axis='y', alpha=0.3)

        # Update title with parameters
        self.fig.suptitle(
            f'{self.title}\n' +
            f'c_int={self.params.c_int:.1f}, c_fail_base={self.params.c_fail_base:.1f}, λ={self.params.lambda_forget:.2f}',
            fontsize=12, weight='bold'
        )

        return self.ax_kitchen, self.ax_memory, self.ax_costs

    def create_animation(self, output_path='kitchen_animation.mp4', fps=10):
        """Create and save animation"""
        print(f"Creating animation: {output_path}")

        # Setup figure
        self.fig = plt.figure(figsize=(14, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

        self.ax_kitchen = self.fig.add_subplot(gs[:, 0])
        self.ax_memory = self.fig.add_subplot(gs[0, 1])
        self.ax_costs = self.fig.add_subplot(gs[1, 1])

        plt.tight_layout()

        # Calculate number of frames
        n_ticks = len(self.history['rewards'])
        n_frames = (n_ticks // self.ticks_per_frame) + 1

        print(f"  Total ticks: {n_ticks}")
        print(f"  Animation frames: {n_frames}")
        print(f"  Duration: {n_frames / fps:.1f} seconds")

        # Create animation
        anim = FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=n_frames,
            interval=1000/fps,  # milliseconds per frame
            blit=False,
            repeat=True
        )

        # Save as video
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"  ✓ Saved to {output_path}")

        plt.close(self.fig)

        return output_path


# ============================================================================
# BATCH VIDEO GENERATION
# ============================================================================

def generate_comparison_videos():
    """Generate videos for different cost configurations"""

    print("\n" + "="*70)
    print("GENERATING KITCHEN VISUALIZATION VIDEOS")
    print("="*70)
    print()

    # Ensure output directory
    output_dir = '/Users/arakawariku/Dropbox/Research/Antti/videos'
    os.makedirs(output_dir, exist_ok=True)

    videos = []

    # ========================================================================
    # Video 1: Low Interruption Cost (Proactive)
    # ========================================================================
    print("Video 1: Low interruption cost (c_int=2) - Proactive Policy")
    print("-" * 70)

    params1 = SimulationParams(c_int=2.0, c_fail_base=20.0, lambda_forget=0.05)
    policy1 = ProactiveReminderPolicy(memory_threshold=0.3)

    result1 = run_simulation(policy1, params1, n_episodes=1, verbose=False)
    history1 = result1['histories'][0]

    print(f"  Episode summary: Reward={sum(history1['rewards']):.1f}, "
          f"Failures={sum(history1['failures'])}, "
          f"Interactions={sum([1 for a in history1['actions_assistant'] if a != 0])}")

    animator1 = KitchenAnimator(
        history1, params1,
        title="Low Cost: Proactive Policy"
    )
    video1 = animator1.create_animation(
        output_path=f'{output_dir}/video1_low_cost_proactive.mp4',
        fps=10
    )
    videos.append(video1)
    print()

    # ========================================================================
    # Video 2: High Interruption Cost (Proactive) - Shows degradation
    # ========================================================================
    print("Video 2: High interruption cost (c_int=15) - Proactive Policy")
    print("-" * 70)

    params2 = SimulationParams(c_int=15.0, c_fail_base=20.0, lambda_forget=0.05)
    policy2 = ProactiveReminderPolicy(memory_threshold=0.3)

    result2 = run_simulation(policy2, params2, n_episodes=1, verbose=False)
    history2 = result2['histories'][0]

    print(f"  Episode summary: Reward={sum(history2['rewards']):.1f}, "
          f"Failures={sum(history2['failures'])}, "
          f"Interactions={sum([1 for a in history2['actions_assistant'] if a != 0])}")

    animator2 = KitchenAnimator(
        history2, params2,
        title="High Cost: Proactive Policy (Degraded)"
    )
    video2 = animator2.create_animation(
        output_path=f'{output_dir}/video2_high_cost_proactive.mp4',
        fps=10
    )
    videos.append(video2)
    print()

    # ========================================================================
    # Video 3: High Interruption Cost (Reactive) - Adapted strategy
    # ========================================================================
    print("Video 3: High interruption cost (c_int=15) - Reactive Policy")
    print("-" * 70)

    params3 = SimulationParams(c_int=15.0, c_fail_base=20.0, lambda_forget=0.05, f0_base=0.35)
    policy3 = ReactivePolicyHighCost(risk_threshold=0.30, params=params3)

    result3 = run_simulation(policy3, params3, n_episodes=1, verbose=False)
    history3 = result3['histories'][0]

    print(f"  Episode summary: Reward={sum(history3['rewards']):.1f}, "
          f"Failures={sum(history3['failures'])}, "
          f"Interactions={sum([1 for a in history3['actions_assistant'] if a != 0])}")

    animator3 = KitchenAnimator(
        history3, params3,
        title="High Cost: Reactive Policy (Adapted)"
    )
    video3 = animator3.create_animation(
        output_path=f'{output_dir}/video3_high_cost_reactive.mp4',
        fps=10
    )
    videos.append(video3)
    print()

    # ========================================================================
    # Video 4: High Failure Cost (Proactive aggressive)
    # ========================================================================
    print("Video 4: High failure cost (c_fail_base=40, c_int=3) - Aggressive Proactive")
    print("-" * 70)

    params4 = SimulationParams(c_int=3.0, c_fail_base=40.0, lambda_forget=0.05)
    policy4 = ProactiveReminderPolicy(memory_threshold=0.25, lookahead=2)

    result4 = run_simulation(policy4, params4, n_episodes=1, verbose=False)
    history4 = result4['histories'][0]

    print(f"  Episode summary: Reward={sum(history4['rewards']):.1f}, "
          f"Failures={sum(history4['failures'])}, "
          f"Interactions={sum([1 for a in history4['actions_assistant'] if a != 0])}")

    animator4 = KitchenAnimator(
        history4, params4,
        title="High Failure Cost: Aggressive Prevention"
    )
    video4 = animator4.create_animation(
        output_path=f'{output_dir}/video4_high_fail_cost.mp4',
        fps=10
    )
    videos.append(video4)
    print()

    # ========================================================================
    # Video 5: Fast Forgetting (needs more reminders)
    # ========================================================================
    print("Video 5: Fast forgetting (λ=0.10) - Proactive Policy")
    print("-" * 70)

    params5 = SimulationParams(c_int=5.0, c_fail_base=20.0, lambda_forget=0.10)
    policy5 = ProactiveReminderPolicy(memory_threshold=0.35, lookahead=1)

    result5 = run_simulation(policy5, params5, n_episodes=1, verbose=False)
    history5 = result5['histories'][0]

    print(f"  Episode summary: Reward={sum(history5['rewards']):.1f}, "
          f"Failures={sum(history5['failures'])}, "
          f"Interactions={sum([1 for a in history5['actions_assistant'] if a != 0])}")

    animator5 = KitchenAnimator(
        history5, params5,
        title="Fast Forgetting: Frequent Reminders"
    )
    video5 = animator5.create_animation(
        output_path=f'{output_dir}/video5_fast_forgetting.mp4',
        fps=10
    )
    videos.append(video5)
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("VIDEO GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated videos:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    print(f"\nAll videos saved to: {output_dir}/")
    print()
    print("Key comparisons:")
    print("  - Videos 1 vs 2: Same policy, different interruption costs")
    print("  - Videos 2 vs 3: Same costs, different policies (Proactive vs Reactive)")
    print("  - Video 4: High failure cost justifies aggressive reminders")
    print("  - Video 5: Fast forgetting requires more frequent intervention")
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate all videos
    generate_comparison_videos()
