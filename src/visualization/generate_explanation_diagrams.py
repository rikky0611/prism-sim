"""
Generate Explanation Diagrams for Procedure Assistant RL Experiment

Creates publication-quality diagrams explaining:
1. System architecture (human-assistant-environment interaction)
2. Task structure (procedural steps with criticality)
3. Memory dynamics (decay and failure probability)
4. Cost structure (regime differences and strategy spaces)

Usage:
    python generate_explanation_diagrams.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from task_definitions import load_task_definitions


def create_system_architecture_diagram():
    """Diagram 1: System architecture showing interaction loop"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Procedure Assistant System Architecture',
           ha='center', fontsize=20, weight='bold')

    # Human Agent box
    human_box = FancyBboxPatch((0.05, 0.4), 0.22, 0.25,
                               boxstyle="round,pad=0.02",
                               facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(human_box)
    ax.text(0.16, 0.58, 'Human Agent', ha='center', fontsize=16, weight='bold')
    ax.text(0.16, 0.52, 'Performing\nProcedural Task',
           ha='center', va='center', fontsize=12)
    ax.text(0.16, 0.44, '• Current step\n• Memory state\n• Responds to reminders',
           ha='center', va='top', fontsize=10)

    # Environment box
    env_box = FancyBboxPatch((0.39, 0.7), 0.22, 0.25,
                            boxstyle="round,pad=0.02",
                            facecolor='lightyellow', edgecolor='black', linewidth=3)
    ax.add_patch(env_box)
    ax.text(0.5, 0.88, 'Environment', ha='center', fontsize=16, weight='bold')
    ax.text(0.5, 0.82, 'Task State',
           ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.74, '• Step progression\n• Memory levels\n• Failure events\n• Elapsed time',
           ha='center', va='top', fontsize=10)

    # Assistant box
    assistant_box = FancyBboxPatch((0.73, 0.4), 0.22, 0.25,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lightgreen', edgecolor='black', linewidth=3)
    ax.add_patch(assistant_box)
    ax.text(0.84, 0.58, 'Procedure\nAssistant (RL)', ha='center', fontsize=16, weight='bold')
    ax.text(0.84, 0.50, 'Decision Making',
           ha='center', va='center', fontsize=12)
    ax.text(0.84, 0.44, '• Observes state\n• Decides: silent/remind\n• Learns from rewards',
           ha='center', va='top', fontsize=10)

    # Reward box (bottom)
    reward_box = FancyBboxPatch((0.39, 0.05), 0.22, 0.15,
                               boxstyle="round,pad=0.02",
                               facecolor='lightcoral', edgecolor='black', linewidth=3)
    ax.add_patch(reward_box)
    ax.text(0.5, 0.15, 'Reward Signal', ha='center', fontsize=14, weight='bold')
    ax.text(0.5, 0.08, '−(failures×c_fail + interruptions×c_int)',
           ha='center', fontsize=11)

    # Arrows - Action: Assistant → Human
    action_arrow = FancyArrowPatch((0.73, 0.52), (0.27, 0.52),
                                  arrowstyle='->', mutation_scale=30,
                                  linewidth=3, color='green')
    ax.add_patch(action_arrow)
    ax.text(0.5, 0.55, 'Action (Silent/Remind)', ha='center',
           fontsize=11, color='darkgreen', weight='bold')

    # Arrows - State Update: Human → Environment
    state_arrow = FancyArrowPatch((0.24, 0.65), (0.42, 0.72),
                                 arrowstyle='->', mutation_scale=30,
                                 linewidth=3, color='blue')
    ax.add_patch(state_arrow)
    ax.text(0.30, 0.70, 'State\nUpdate', ha='center',
           fontsize=10, color='darkblue', weight='bold')

    # Arrows - Observation: Environment → Assistant
    obs_arrow = FancyArrowPatch((0.58, 0.72), (0.76, 0.65),
                               arrowstyle='->', mutation_scale=30,
                               linewidth=3, color='purple')
    ax.add_patch(obs_arrow)
    ax.text(0.70, 0.70, 'Observation\n(Partial)', ha='center',
           fontsize=10, color='purple', weight='bold')

    # Arrows - Reward: Environment → Assistant
    reward_arrow1 = FancyArrowPatch((0.5, 0.20), (0.5, 0.40),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='red', linestyle='dashed')
    ax.add_patch(reward_arrow1)

    reward_arrow2 = FancyArrowPatch((0.61, 0.20), (0.82, 0.40),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='red', linestyle='dashed')
    ax.add_patch(reward_arrow2)
    ax.text(0.72, 0.28, 'Reward', ha='center',
           fontsize=11, color='darkred', weight='bold')

    output_path = PROJECT_ROOT / "results" / "figures" / "explanation_1_system_architecture.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_task_structure_diagram():
    """Diagram 2: Task structure with step criticality"""
    tasks = load_task_definitions()

    # Create 2 subplots for 2 example tasks
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle('Procedural Task Structure with Step Criticality', fontsize=20, weight='bold', y=0.98)

    # Task 1: make_cereal (simple, 8 steps)
    task1 = tasks['make_cereal']
    plot_task_structure(ax1, task1, 'make_cereal (8 steps - Simple Task)')

    # Task 2: latte_making (complex, 20 steps)
    task2 = tasks['latte_making']
    plot_task_structure(ax2, task2, 'latte_making (20 steps - Complex Task)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', label='Trivial (criticality=0.0)'),
        Patch(facecolor='orange', edgecolor='black', label='Critical (criticality=1.0)'),
        Patch(facecolor='red', edgecolor='black', label='Ultra-Critical (criticality>1.0)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, frameon=True)

    output_path = PROJECT_ROOT / "results" / "figures" / "explanation_2_task_structure.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_task_structure(ax, task_def, title):
    """Helper to plot a single task structure"""
    ax.set_xlim(-0.5, len(task_def.steps) + 0.5)
    ax.set_ylim(-1.5, 3)
    ax.axis('off')
    ax.set_title(title, fontsize=16, weight='bold', pad=10)

    for i, step in enumerate(task_def.steps):
        x = i
        y = 1

        # Color by criticality
        if step.criticality == 0:
            color = 'lightgray'
        elif step.criticality <= 1.0:
            color = 'orange'
        else:
            color = 'red'

        # Draw step circle
        circle = Circle((x, y), 0.35, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Step number
        ax.text(x, y + 0.15, f'{i}', ha='center', va='center', fontsize=10, weight='bold')

        # Step name (abbreviated)
        name = step.name.replace('_', ' ').title()
        if len(name) > 12:
            name = name[:10] + '...'
        ax.text(x, y - 0.15, name, ha='center', va='center', fontsize=8)

        # Criticality value below
        ax.text(x, y - 0.6, f'Crit: {step.criticality:.1f}',
               ha='center', fontsize=9, style='italic')

        # Failure cost below that
        cost = task_def.base_failure_cost * step.criticality
        ax.text(x, y - 0.9, f'Cost: {cost:.0f}',
               ha='center', fontsize=9, color='darkred')

        # Arrow to next step
        if i < len(task_def.steps) - 1:
            arrow = FancyArrowPatch((x + 0.37, y), (x + 0.63, y),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color='gray')
            ax.add_patch(arrow)


def create_memory_dynamics_diagram():
    """Diagram 3: Memory decay and failure probability"""
    # Simulate memory dynamics
    ticks = np.arange(0, 100)
    memory = np.zeros(len(ticks))

    # Initial memory
    memory[0] = 0.8

    # Parameters (from actual simulation)
    lambda_forget = 0.03  # 23-tick half-life (slower decay)
    delta_reminder = 0.8  # Memory restoration from reminder (stronger boost)
    f0_base = 0.6  # 60% baseline failure rate when m=0
    k_memory = 3.0  # Memory effect steepness

    # Intervention at tick 30
    intervention_tick = 30

    for t in range(1, len(ticks)):
        # Memory decay: m_t+1 = m_t × (1 - λ_forget)
        memory[t] = memory[t-1] * (1 - lambda_forget)

        # Intervention: memory boost
        if t == intervention_tick:
            memory[t] = min(memory[t] + delta_reminder, 2.0)

    # Compute failure probability: f(m) = f0_base × exp(-k × m)
    fail_prob = f0_base * np.exp(-k_memory * memory)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Memory Dynamics and Failure Probability', fontsize=20, weight='bold')

    # Top: Memory evolution
    ax1.plot(ticks, memory, label='Memory Level', linewidth=3, color='blue')
    ax1.axvline(intervention_tick, color='red', linestyle=':', linewidth=2,
               label='Intervention (Reminder)')
    ax1.axhline(0.3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Low Memory Threshold')

    ax1.set_ylabel('Memory Level', fontsize=14, weight='bold')
    ax1.set_xlabel('Time (ticks)', fontsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.5)

    # Annotations
    ax1.annotate('Memory restored\nby reminder\n(+0.8 boost)',
                xy=(intervention_tick, memory[intervention_tick]),
                xytext=(intervention_tick + 15, memory[intervention_tick] + 0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, weight='bold')

    ax1.annotate('Exponential decay\n(λ=0.03, 23-tick half-life)',
                xy=(60, memory[60]),
                xytext=(60, memory[60] + 0.25),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)

    # Bottom: Failure probability
    ax2.plot(ticks, fail_prob, linewidth=3, color='darkred', label='Failure Probability')
    ax2.axvline(intervention_tick, color='red', linestyle=':', linewidth=2)
    ax2.fill_between(ticks, 0, fail_prob, alpha=0.3, color='red')

    ax2.set_xlabel('Time (ticks)', fontsize=14, weight='bold')
    ax2.set_ylabel('Failure Probability', fontsize=14, weight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.7)

    # Formula annotation
    formula_text = r'$f(m) = f_0 \times e^{-k \cdot m}$'
    ax2.text(0.98, 0.95, formula_text, transform=ax2.transAxes,
            fontsize=14, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add example values
    ax2.text(0.98, 0.85, f'f₀ = {f0_base} (baseline), k = {k_memory} (steepness)',
            transform=ax2.transAxes, fontsize=10, ha='right', va='top',
            style='italic')

    # Add concrete examples
    examples_text = 'Examples:\n' \
                   f'm=0.0 → f={f0_base:.0%}\n' \
                   f'm=0.3 → f={f0_base * np.exp(-k_memory * 0.3):.0%}\n' \
                   f'm=0.6 → f={f0_base * np.exp(-k_memory * 0.6):.0%}\n' \
                   f'm=1.0 → f={f0_base * np.exp(-k_memory * 1.0):.0%}'
    ax2.text(0.02, 0.95, examples_text, transform=ax2.transAxes,
            fontsize=9, ha='left', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    output_path = PROJECT_ROOT / "results" / "figures" / "explanation_3_memory_dynamics.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_cost_strategy_diagram():
    """Diagram 4: Cost structure and strategy spaces"""
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle('Cost Regimes and Optimal Strategy Regions', fontsize=20, weight='bold')

    # Left: Cost regime table
    regimes_data = [
        ['Extremely High', '50', '1', '50:1', 'Very rare interventions'],
        ['Moderate', '15', '1', '15:1', 'Mixed strategy'],
        ['Extremely Low', '5', '1', '5:1', 'Strategic silence']
    ]

    table = ax1.table(cellText=regimes_data,
                     colLabels=['Regime', 'c_fail', 'c_int', 'Ratio', 'Expected Behavior'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.12, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style regime rows
    colors = ['#FFE6E6', '#FFF4E6', '#E6F7FF']
    for i, color in enumerate(colors, start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(color)

    ax1.axis('off')
    ax1.set_title('Cost Regime Configuration', fontsize=16, weight='bold', pad=20)

    # Right: Strategy heatmap
    # Decision boundary: intervene if p_fail > c_int / c_fail
    fail_probs = np.linspace(0, 1, 100)
    cost_ratios = np.linspace(1, 50, 100)

    strategy_map = np.zeros((len(cost_ratios), len(fail_probs)))
    for i, ratio in enumerate(cost_ratios):
        for j, p_fail in enumerate(fail_probs):
            threshold = 1.0 / ratio  # c_int / c_fail
            if p_fail > threshold:
                strategy_map[i, j] = 1  # Intervene (red)
            else:
                strategy_map[i, j] = 0  # Silent (blue)

    # Plot heatmap
    sns.heatmap(strategy_map, ax=ax2, cmap=['#4472C4', '#FF6B6B'],
               cbar_kws={'label': 'Optimal Strategy', 'ticks': [0.25, 0.75]},
               xticklabels=False, yticklabels=False)

    # Customize colorbar
    cbar = ax2.collections[0].colorbar
    cbar.set_ticklabels(['Silent', 'Intervene'])
    cbar.ax.tick_params(labelsize=12)

    # Add regime markers
    regime_markers = [
        (30, 'Very High\nStakes', '#8B0000'),
        (15, 'Balanced', '#006400'),
        (10, 'Moderate\nLow', '#00008B')
    ]

    for ratio, label, color in regime_markers:
        y_pos = int((ratio - 1) / (50 - 1) * 99)
        ax2.axhline(y=y_pos, color=color, linestyle='--', linewidth=2.5, alpha=0.8)
        ax2.text(102, y_pos, label, fontsize=11, weight='bold',
                color=color, va='center')

    # Set labels
    ax2.set_xlabel('Failure Probability', fontsize=14, weight='bold')
    ax2.set_ylabel('Cost Ratio (c_fail / c_int)', fontsize=14, weight='bold')
    ax2.set_title('Optimal Decision Boundary', fontsize=16, weight='bold', pad=10)

    # Add axis labels
    ax2.set_xticks([0, 25, 50, 75, 99])
    ax2.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    ax2.set_yticks([0, 25, 50, 75, 99])
    ax2.set_yticklabels(['1', '13', '25', '37', '50'])

    output_path = PROJECT_ROOT / "results" / "figures" / "explanation_4_cost_strategy.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Generate all explanation diagrams"""
    print("=" * 80)
    print("GENERATING EXPLANATION DIAGRAMS")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate diagrams
    print("1. System Architecture...")
    create_system_architecture_diagram()

    print("2. Task Structure...")
    create_task_structure_diagram()

    print("3. Memory Dynamics...")
    create_memory_dynamics_diagram()

    print("4. Cost Strategy...")
    create_cost_strategy_diagram()

    print()
    print("=" * 80)
    print("✓ ALL DIAGRAMS GENERATED")
    print("=" * 80)
    print()
    print(f"Output location: {output_dir}")
    print()
    print("Files created:")
    print("  • explanation_1_system_architecture.png")
    print("  • explanation_2_task_structure.png")
    print("  • explanation_3_memory_dynamics.png")
    print("  • explanation_4_cost_strategy.png")


if __name__ == "__main__":
    main()
