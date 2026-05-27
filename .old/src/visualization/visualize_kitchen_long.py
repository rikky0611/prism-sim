"""
Extended Kitchen Visualization - Longer Videos for Better Interpretation

Creates longer, slower-paced videos showing:
- Extended episode durations (longer step times)
- Slower animation (1 tick per frame instead of 2)
- More detail and clarity for interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from procedure_assistant_sim import *
from visualize_kitchen import KitchenLayout
import os


class ExtendedKitchenAnimator:
    """Generate extended animated videos with better pacing"""

    def __init__(self, history, params, title="Procedure Assistant"):
        self.history = history
        self.params = params
        self.title = title
        self.layout = KitchenLayout()

        # Slower animation for better interpretation
        self.frame_rate = 15  # frames per second
        self.ticks_per_frame = 1  # Show every simulation tick

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

        # Draw human agent with pulsing effect
        agent_pos = self.get_agent_position(tick)
        pulse = 0.05 * np.sin(frame_num * 0.3)  # Subtle pulse animation
        agent_circle = patches.Circle(
            agent_pos, 0.3 + pulse,
            facecolor=self.layout.colors['human'],
            edgecolor='white', linewidth=2, zorder=10
        )
        self.ax_kitchen.add_patch(agent_circle)

        # Add agent label
        self.ax_kitchen.text(
            agent_pos[0], agent_pos[1] - 0.6,
            'Human',
            ha='center', fontsize=8, weight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=self.layout.colors['human'], alpha=0.8)
        )

        # Draw assistant notification if active (with animation)
        if assistant_action != ASSISTANT_ACTIONS['silent']:
            notify_x, notify_y = agent_pos[0] + 1.5, agent_pos[1] + 0.8

            # Animated bubble (slight scale)
            scale = 1.0 + 0.1 * np.sin(frame_num * 0.5)

            bubble = patches.FancyBboxPatch(
                (notify_x - 1.0 * scale, notify_y - 0.5 * scale),
                2.0 * scale, 1.0 * scale,
                boxstyle="round,pad=0.15",
                facecolor=self.layout.colors['assistant'],
                edgecolor='white', linewidth=3, zorder=15
            )
            self.ax_kitchen.add_patch(bubble)

            # Text
            if assistant_action == ASSISTANT_ACTIONS['confirm']:
                text = "❓ Confirm?\nWhat step?"
                fontsize = 8
            else:
                # It's a reminder
                for i in range(len(PROCEDURAL_STEPS)):
                    if assistant_action == ASSISTANT_ACTIONS[f'remind_{i}']:
                        step_text = PROCEDURAL_STEPS[i].replace('_', ' ').title()
                        text = f"💡 Reminder:\n{step_text}"
                        fontsize = 7
                        break
                else:
                    text = "💡 Reminder"
                    fontsize = 8

            self.ax_kitchen.text(
                notify_x, notify_y, text,
                ha='center', va='center', fontsize=fontsize,
                color='white', weight='bold', zorder=16
            )

            # Add "ASSISTANT" label
            self.ax_kitchen.text(
                notify_x, notify_y - 0.8,
                'ASSISTANT',
                ha='center', fontsize=6, weight='bold',
                color=self.layout.colors['assistant'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )

        # Draw failure effect (animated)
        if failure:
            # Red X over agent
            x, y = agent_pos
            offset = 0.5 + 0.1 * np.sin(frame_num * 0.8)
            self.ax_kitchen.plot(
                [x-offset, x+offset], [y-offset, y+offset],
                'r-', linewidth=5, zorder=12
            )
            self.ax_kitchen.plot(
                [x-offset, x+offset], [y+offset, y-offset],
                'r-', linewidth=5, zorder=12
            )

            # Failure text
            self.ax_kitchen.text(
                x, y + 0.8,
                '⚠️ FAILURE',
                ha='center', fontsize=10, weight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.9)
            )

        # Draw current step indicator with progress
        if step_idx < len(PROCEDURAL_STEPS):
            step_text = f"Step {step_idx+1}/5: {PROCEDURAL_STEPS[step_idx].replace('_', ' ').title()}"
            progress = f"(Tick {tau})"
        else:
            step_text = "✅ TASK COMPLETE"
            progress = ""

        self.ax_kitchen.text(
            self.layout.width / 2, 0.5,
            f"{step_text}\n{progress}",
            ha='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2)
        )

        # Draw step progression indicator
        step_progress = (step_idx + 1) / len(PROCEDURAL_STEPS)
        progress_bar_width = self.layout.width * 0.8
        progress_bar_x = (self.layout.width - progress_bar_width) / 2
        progress_bar_y = 7.2

        # Background bar
        self.ax_kitchen.add_patch(patches.Rectangle(
            (progress_bar_x, progress_bar_y), progress_bar_width, 0.3,
            facecolor='lightgray', edgecolor='black', linewidth=1, zorder=5
        ))

        # Progress bar
        if step_idx <= len(PROCEDURAL_STEPS):
            filled_width = progress_bar_width * step_progress
            self.ax_kitchen.add_patch(patches.Rectangle(
                (progress_bar_x, progress_bar_y), filled_width, 0.3,
                facecolor='green', edgecolor='black', linewidth=1, zorder=6
            ))

        # Update memory bars with more detail
        memory = obs['memory']
        self.ax_memory.clear()

        step_names = [s.replace('_', '\n') for s in PROCEDURAL_STEPS]
        y_positions = np.arange(len(PROCEDURAL_STEPS))

        # Color bars based on memory level
        colors = []
        for i, m in enumerate(memory):
            if i == step_idx:
                colors.append('darkgreen')  # Current step
            elif m < 0.3:
                colors.append('red')  # Low memory - danger!
            elif m < 0.5:
                colors.append('orange')  # Medium memory
            else:
                colors.append('lightgreen')  # Good memory

        bars = self.ax_memory.barh(y_positions, memory, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, memory)):
            self.ax_memory.text(
                val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}',
                va='center', fontsize=8, weight='bold'
            )

        self.ax_memory.set_yticks(y_positions)
        self.ax_memory.set_yticklabels(step_names, fontsize=8)
        self.ax_memory.set_xlim(0, 1.0)
        self.ax_memory.set_xlabel('Memory Level', fontsize=10, weight='bold')
        self.ax_memory.set_title('Memory State', fontsize=11, weight='bold')

        # Add threshold lines
        self.ax_memory.axvline(0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Low threshold')
        self.ax_memory.axvline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Medium')

        self.ax_memory.legend(loc='upper right', fontsize=7)
        self.ax_memory.grid(axis='x', alpha=0.3)

        # Update cost tracking with more detail
        cumulative_reward = sum(self.history['rewards'][:tick+1])
        total_failures = sum(self.history['failures'][:tick+1])
        total_interactions = sum([1 for a in self.history['actions_assistant'][:tick+1]
                                   if a != ASSISTANT_ACTIONS['silent']])

        failure_cost = -total_failures * self.params.c_fail_base
        interaction_cost = -total_interactions * self.params.c_int

        self.ax_costs.clear()

        # Cost breakdown
        cost_labels = ['Total\nReward', 'Failure\nCost', 'Interrupt\nCost']
        cost_values = [cumulative_reward, failure_cost, interaction_cost]
        colors = ['blue' if v > 0 else 'red' for v in cost_values]

        bars = self.ax_costs.bar(cost_labels, cost_values, color=colors, alpha=0.8,
                                 edgecolor='black', linewidth=2)

        # Add value labels on bars
        for bar, val in zip(bars, cost_values):
            height = bar.get_height()
            self.ax_costs.text(
                bar.get_x() + bar.get_width()/2.,
                height + (5 if height > 0 else -5),
                f'{val:.0f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, weight='bold'
            )

        self.ax_costs.axhline(0, color='black', linewidth=2)
        self.ax_costs.set_ylabel('Cost', fontsize=10, weight='bold')
        self.ax_costs.set_title(f'Costs at Tick {tick}/{len(self.history["rewards"])-1}',
                                fontsize=11, weight='bold')
        self.ax_costs.grid(axis='y', alpha=0.3)

        # Add count labels at bottom
        count_text = f'Failures: {total_failures} | Interactions: {total_interactions}'
        self.ax_costs.text(
            0.5, -0.15, count_text,
            ha='center', transform=self.ax_costs.transAxes,
            fontsize=9, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )

        # Update title with parameters
        self.fig.suptitle(
            f'{self.title}\n' +
            f'c_int={self.params.c_int:.1f}, c_fail={self.params.c_fail_base:.1f}, ' +
            f'λ={self.params.lambda_forget:.3f}',
            fontsize=13, weight='bold'
        )

        return self.ax_kitchen, self.ax_memory, self.ax_costs

    def create_animation(self, output_path='kitchen_animation_long.mp4', fps=15):
        """Create and save animation"""
        print(f"Creating extended animation: {output_path}")

        # Setup figure
        self.fig = plt.figure(figsize=(16, 7))
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
            interval=1000/fps,
            blit=False,
            repeat=True
        )

        # Save as video with higher quality
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(output_path, writer=writer)
        print(f"  ✓ Saved to {output_path}")

        plt.close(self.fig)

        return output_path


def generate_extended_videos():
    """Generate longer, more interpretable videos"""

    print("\n" + "="*70)
    print("GENERATING EXTENDED KITCHEN VISUALIZATION VIDEOS")
    print("="*70)
    print("\nThese videos have:")
    print("  - Longer episodes (extended step durations)")
    print("  - Slower animation (1 tick per frame)")
    print("  - More visual detail and annotations")
    print("  - Better for interpretation and presentations")
    print()

    output_dir = '/Users/arakawariku/Dropbox/Research/Antti/videos'
    os.makedirs(output_dir, exist_ok=True)

    videos = []

    # Use extended parameters for longer episodes
    extended_params_base = {
        'step_mean_duration': 60,  # 2× longer steps
        'step_std_duration': 15,
    }

    # ========================================================================
    # Video 1: Low Interruption Cost - Extended
    # ========================================================================
    print("Video 1: Low interruption cost (c_int=2) - Proactive Policy [EXTENDED]")
    print("-" * 70)

    params1 = SimulationParams(
        c_int=2.0, c_fail_base=20.0, lambda_forget=0.05,
        **extended_params_base
    )
    policy1 = ProactiveReminderPolicy(memory_threshold=0.3)

    result1 = run_simulation(policy1, params1, n_episodes=1, verbose=False)
    history1 = result1['histories'][0]

    print(f"  Episode: Reward={sum(history1['rewards']):.1f}, "
          f"Failures={sum(history1['failures'])}, "
          f"Interactions={sum([1 for a in history1['actions_assistant'] if a != 0])}, "
          f"Length={len(history1['rewards'])} ticks")

    animator1 = ExtendedKitchenAnimator(
        history1, params1,
        title="Scenario 1: Low Interruption Cost - Proactive Works Well"
    )
    video1 = animator1.create_animation(
        output_path=f'{output_dir}/extended_video1_low_cost.mp4',
        fps=15
    )
    videos.append(video1)
    print()

    # ========================================================================
    # Video 2: High Interruption Cost (Proactive) - Extended
    # ========================================================================
    print("Video 2: High interruption cost (c_int=15) - Proactive Policy [EXTENDED]")
    print("-" * 70)

    params2 = SimulationParams(
        c_int=15.0, c_fail_base=20.0, lambda_forget=0.05,
        **extended_params_base
    )
    policy2 = ProactiveReminderPolicy(memory_threshold=0.3)

    result2 = run_simulation(policy2, params2, n_episodes=1, verbose=False)
    history2 = result2['histories'][0]

    print(f"  Episode: Reward={sum(history2['rewards']):.1f}, "
          f"Failures={sum(history2['failures'])}, "
          f"Interactions={sum([1 for a in history2['actions_assistant'] if a != 0])}, "
          f"Length={len(history2['rewards'])} ticks")

    animator2 = ExtendedKitchenAnimator(
        history2, params2,
        title="Scenario 2: High Interruption Cost - Proactive Fails (Watch Costs Explode!)"
    )
    video2 = animator2.create_animation(
        output_path=f'{output_dir}/extended_video2_high_cost_proactive.mp4',
        fps=15
    )
    videos.append(video2)
    print()

    # ========================================================================
    # Video 3: High Interruption Cost (Reactive) - Extended
    # ========================================================================
    print("Video 3: High interruption cost (c_int=15) - Reactive Policy [EXTENDED]")
    print("-" * 70)

    params3 = SimulationParams(
        c_int=15.0, c_fail_base=20.0, lambda_forget=0.05, f0_base=0.35,
        **extended_params_base
    )
    policy3 = ReactivePolicyHighCost(risk_threshold=0.30, params=params3)

    result3 = run_simulation(policy3, params3, n_episodes=1, verbose=False)
    history3 = result3['histories'][0]

    print(f"  Episode: Reward={sum(history3['rewards']):.1f}, "
          f"Failures={sum(history3['failures'])}, "
          f"Interactions={sum([1 for a in history3['actions_assistant'] if a != 0])}, "
          f"Length={len(history3['rewards'])} ticks")

    animator3 = ExtendedKitchenAnimator(
        history3, params3,
        title="Scenario 3: High Interruption Cost - Reactive Adapts (Fewer Interrupts)"
    )
    video3 = animator3.create_animation(
        output_path=f'{output_dir}/extended_video3_high_cost_reactive.mp4',
        fps=15
    )
    videos.append(video3)
    print()

    # ========================================================================
    # Video 4: Fast Forgetting - Extended
    # ========================================================================
    print("Video 4: Fast forgetting (λ=0.10) - Watch Memory Decay! [EXTENDED]")
    print("-" * 70)

    params4 = SimulationParams(
        c_int=5.0, c_fail_base=20.0, lambda_forget=0.10,
        **extended_params_base
    )
    policy4 = ProactiveReminderPolicy(memory_threshold=0.35, lookahead=1)

    result4 = run_simulation(policy4, params4, n_episodes=1, verbose=False)
    history4 = result4['histories'][0]

    print(f"  Episode: Reward={sum(history4['rewards']):.1f}, "
          f"Failures={sum(history4['failures'])}, "
          f"Interactions={sum([1 for a in history4['actions_assistant'] if a != 0])}, "
          f"Length={len(history4['rewards'])} ticks")

    animator4 = ExtendedKitchenAnimator(
        history4, params4,
        title="Scenario 4: Fast Forgetting - Frequent Reminders Needed (Memory Decays Fast)"
    )
    video4 = animator4.create_animation(
        output_path=f'{output_dir}/extended_video4_fast_forgetting.mp4',
        fps=15
    )
    videos.append(video4)
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("EXTENDED VIDEO GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated videos:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")

    print(f"\nAll videos saved to: {output_dir}/")
    print()
    print("Video characteristics:")
    print("  - 2× longer episodes (step duration: 60 ticks vs 30)")
    print("  - Smoother animation (1 tick/frame vs 2 ticks/frame)")
    print("  - Enhanced visual details (animations, labels, progress bar)")
    print("  - Duration: 5-10 seconds per video (vs 1.5-3 seconds)")
    print()
    print("Key observations to watch for:")
    print("  - Video 1: Notice moderate reminder frequency, good memory levels")
    print("  - Video 2: Watch interaction cost bar grow rapidly!")
    print("  - Video 3: Notice fewer assistant notifications, better total cost")
    print("  - Video 4: Watch memory bars decay quickly, many reminders needed")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    generate_extended_videos()
