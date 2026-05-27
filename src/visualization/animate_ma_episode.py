"""
Animated Multi-Agent Episode Visualization

Creates publication-quality MP4/GIF videos showing emergent communication
behaviors (narration, question) during procedural task execution.

Features:
  - Task graph panel: DAG for non-linear tasks, styled chain for linear
  - Action timeline: swim lanes for human + assistant with colored markers
  - State panels: step belief, memory levels, obs noise, cumulative reward
  - Supports all 7 tasks and both standard + probe configurations

Usage:
  python3 src/visualization/animate_ma_episode.py \\
      --task make_tea --regime extremely_high --probe A --seed 42

Output:
  results/videos/ma_episode_{task}_{regime}[_probeX].mp4  (or .gif fallback)
"""

import sys
import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions, get_task_definition, TaskDefinition
from ma_procedure_assistant_sim import MAProcedureAssistantEnv, MASimulationParams
from train_ma_ippo import define_ma_cost_regimes, HumanGymWrapper, AssistantGymWrapper
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# COLORS (colorblind-safe)
# ---------------------------------------------------------------------------
C_NARRATE = '#0173b2'
C_QUESTION = '#029e73'
C_REMIND = '#de8f05'
C_CONFIRM = '#cc78bc'
C_SILENT = '#cccccc'
C_TRIVIAL = '#d0d0d0'
C_CRITICAL = '#de8f05'
C_ULTRA = '#d55e00'
C_COMPLETED = '#029e73'
C_FAILED = '#d55e00'
C_BG = '#fafafa'

# Probe definitions (mirrored from run_narration_probe.py)
PROBES = {
    'A': {'param_overrides': {'c_nar': 0.1, 'c_q': 2.0}},
    'B': {'param_overrides': {'obs_noise_min': 0.001}},
}


# ============================================================================
# TickRecord
# ============================================================================
@dataclass
class TickRecord:
    tick: int
    true_step: int          # position index (0..N-1 or N if done)
    true_identity: int      # identity at this position
    h_action: int
    a_action: int
    step_completed: bool
    failure: bool
    memory: np.ndarray
    obs_noise_state: float
    step_belief: np.ndarray
    cumulative_reward: float
    episode_order: np.ndarray


# ============================================================================
# TaskDAGLayout
# ============================================================================
class TaskDAGLayout:
    """Derive a DAG from rollout_patterns and compute node positions."""

    def __init__(self, task_def: TaskDefinition):
        self.task_def = task_def
        self.n = task_def.n_steps
        self.G = self._derive_dag()
        self.pos = self._compute_layout()

    def _derive_dag(self) -> nx.DiGraph:
        n = self.n
        patterns = self.task_def.rollout_patterns
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        if not patterns:
            # Linear: chain 0→1→2→...
            for i in range(n - 1):
                G.add_edge(i, i + 1)
            return G

        # Compute partial order: a before b iff a precedes b in ALL patterns
        before = np.ones((n, n), dtype=bool)
        for p in patterns:
            pos_map = {v: i for i, v in enumerate(p)}
            for a in range(n):
                for b in range(n):
                    if pos_map[a] >= pos_map[b]:
                        before[a, b] = False

        for a in range(n):
            for b in range(n):
                if a != b and before[a, b]:
                    G.add_edge(a, b)

        G = nx.transitive_reduction(G)
        return G

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """Layered layout: x = layer, y = position within layer."""
        layers = list(nx.topological_generations(self.G))
        pos = {}
        max_layer_size = max(len(layer) for layer in layers)
        for layer_idx, layer in enumerate(layers):
            layer_sorted = sorted(layer)
            for i, node in enumerate(layer_sorted):
                x = layer_idx
                # Center nodes within layer
                y = (i - (len(layer_sorted) - 1) / 2.0) * 1.2
                pos[node] = (x, y)
        return pos


# ============================================================================
# EpisodeRecorder
# ============================================================================
class EpisodeRecorder:
    """Run a live episode with trained models and record per-tick state."""

    def run_episode(
        self,
        task_name: str,
        regime_name: str,
        probe_id: Optional[str] = None,
        seed: int = 42,
        deterministic: bool = False,
    ) -> Tuple[List[TickRecord], MAProcedureAssistantEnv, MASimulationParams]:
        task_def = get_task_definition(task_name)

        # Build params
        regimes = define_ma_cost_regimes(task_def)
        base_params = regimes[regime_name]
        if probe_id and probe_id in PROBES:
            overrides = PROBES[probe_id]['param_overrides']
            params = MASimulationParams(
                lambda_forget=base_params.lambda_forget,
                delta_reminder=base_params.delta_reminder,
                f0_base=base_params.f0_base,
                k_memory=base_params.k_memory,
                c_fail_per_step=base_params.c_fail_per_step,
                c_remind=base_params.c_remind,
                c_nar=overrides.get('c_nar', base_params.c_nar),
                c_off_timing=base_params.c_off_timing,
                obs_noise=base_params.obs_noise,
                obs_noise_min=overrides.get('obs_noise_min', base_params.obs_noise_min),
                lambda_noise_recover=overrides.get('lambda_noise_recover', base_params.lambda_noise_recover),
                delta_q=overrides.get('delta_q', base_params.delta_q),
                c_q=overrides.get('c_q', base_params.c_q),
                c_confirm=overrides.get('c_confirm', base_params.c_confirm),
            )
        else:
            params = base_params

        ma_env = MAProcedureAssistantEnv(params, task_def)

        # Load trained models
        model_dir = PROJECT_ROOT / "models" / "ma_ippo" / task_name / regime_name
        human_model = PPO.load(str(model_dir / "human_model_best"))
        assistant_model = PPO.load(str(model_dir / "assistant_model_best"))

        # Run episode
        np.random.seed(seed)
        random.seed(seed)
        h_obs_dict, a_obs_dict = ma_env.reset()

        # Obs conversion helpers (reuse wrapper logic)
        def convert_h(obs_dict):
            return np.array([
                obs_dict['current_identity'], obs_dict['tau'],
                obs_dict['memory_current'], obs_dict['assistant_last_action'],
                obs_dict['obs_noise_state'],
            ], dtype=np.float32)

        def convert_a(obs_dict):
            return np.concatenate([
                obs_dict['step_belief'].astype(np.float32),
                obs_dict['expected_tau'].astype(np.float32),
                obs_dict['memory_estimate_critical'].astype(np.float32),
                np.array([obs_dict['human_last_action']], dtype=np.float32),
            ])

        records = []
        cum_reward = 0.0

        while not ma_env.ma_state.is_done:
            h_obs = convert_h(h_obs_dict)
            a_obs = convert_a(a_obs_dict)
            h_action, _ = human_model.predict(h_obs, deterministic=deterministic)
            a_action, _ = assistant_model.predict(a_obs, deterministic=deterministic)

            pre_belief = ma_env.ma_state.step_belief.copy()
            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(
                int(h_action), int(a_action)
            )
            cum_reward += reward

            true_step = info['true_step']
            if true_step < ma_env.n_steps:
                true_id = int(ma_env.episode_order[true_step])
            else:
                true_id = ma_env.n_steps  # done sentinel

            records.append(TickRecord(
                tick=len(records),
                true_step=true_step,
                true_identity=true_id,
                h_action=int(h_action),
                a_action=int(a_action),
                step_completed=info['step_completed'],
                failure=info['failure'],
                memory=info['memory'].copy(),
                obs_noise_state=info['obs_noise_state'],
                step_belief=ma_env.ma_state.step_belief.copy(),
                cumulative_reward=cum_reward,
                episode_order=ma_env.episode_order.copy(),
            ))

        return records, ma_env, params


# ============================================================================
# MAEpisodeAnimator
# ============================================================================
class MAEpisodeAnimator:
    """Create animated visualization of a multi-agent episode."""

    def __init__(
        self,
        records: List[TickRecord],
        ma_env: MAProcedureAssistantEnv,
        params: MASimulationParams,
        task_name: str,
        regime_name: str,
        probe_id: Optional[str] = None,
    ):
        self.records = records
        self.ma_env = ma_env
        self.params = params
        self.task_def = ma_env.task_def
        self.task_name = task_name
        self.regime_name = regime_name
        self.probe_id = probe_id
        self.n_steps = ma_env.n_steps
        self.dag = TaskDAGLayout(self.task_def)

        # Action name maps
        self.h_action_names, self.a_action_names = ma_env.get_action_names()
        # Critical step indices
        self.critical_steps = ma_env.critical_steps
        # Reverse maps for question/remind targets
        self._question_id_to_step = ma_env._question_id_to_step
        self._remind_id_to_step = ma_env._remind_id_to_step

        # Build frame list with pacing
        self._build_frame_map()

    def _build_frame_map(self):
        """Map frame indices to tick indices with freeze frames."""
        self.frame_to_tick = []
        fps = 4
        # Freeze at start
        for _ in range(fps * 2):
            self.frame_to_tick.append(0)
        # Main ticks
        for i, rec in enumerate(self.records):
            self.frame_to_tick.append(i)
            if rec.step_completed:
                # Extra frames on step completion
                self.frame_to_tick.append(i)
                self.frame_to_tick.append(i)
        # Freeze at end
        last = len(self.records) - 1
        for _ in range(fps * 3):
            self.frame_to_tick.append(last)
        self.total_frames = len(self.frame_to_tick)

    def _setup_figure(self):
        """Create figure with 5-panel layout."""
        self.fig = plt.figure(figsize=(19.2, 10.8), facecolor='white')
        gs = gridspec.GridSpec(
            3, 3, figure=self.fig,
            width_ratios=[1.1, 1.3, 0.8],
            height_ratios=[1, 1, 0.8],
            hspace=0.35, wspace=0.3,
            left=0.05, right=0.97, top=0.90, bottom=0.06,
        )

        self.ax_graph = self.fig.add_subplot(gs[:, 0])
        self.ax_timeline = self.fig.add_subplot(gs[:, 1])
        self.ax_belief = self.fig.add_subplot(gs[0, 2])
        self.ax_memory = self.fig.add_subplot(gs[1, 2])
        self.ax_gauges = self.fig.add_subplot(gs[2, 2])

        for ax in [self.ax_graph, self.ax_timeline, self.ax_belief,
                    self.ax_memory, self.ax_gauges]:
            ax.set_facecolor(C_BG)
            for spine in ax.spines.values():
                spine.set_color('#cccccc')

        # Title
        probe_str = f"Probe {self.probe_id}" if self.probe_id else "Standard"
        self.title_text = self.fig.suptitle(
            f"Task: {self.task_name} ({self.n_steps} steps)  |  "
            f"Regime: {self.regime_name}  |  {probe_str}  |  "
            f"Tick: 0 / {len(self.records)}",
            fontsize=14, fontweight='bold', fontfamily='monospace',
            y=0.96,
        )

    # ------------------------------------------------------------------
    # Panel A: Task Graph
    # ------------------------------------------------------------------
    def _init_graph(self):
        ax = self.ax_graph
        ax.set_title("Task Graph", fontsize=11, fontweight='bold', pad=8)
        ax.set_aspect('equal')
        ax.axis('off')

        G = self.dag.G
        pos = self.dag.pos
        step_names = self.task_def.step_names

        # Normalize positions to [0, 1] range
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 1)
        y_range = max(y_max - y_min, 1)

        self.node_pos_norm = {}
        for node, (x, y) in pos.items():
            nx_ = 0.1 + 0.8 * (x - x_min) / x_range
            ny_ = 0.2 + 0.6 * (y - y_min) / y_range if y_range > 0.01 else 0.5
            self.node_pos_norm[node] = (nx_, ny_)

        # Draw edges
        for u, v in G.edges():
            x0, y0 = self.node_pos_norm[u]
            x1, y1 = self.node_pos_norm[v]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color='#888888', lw=1.5,
                    connectionstyle='arc3,rad=0.1',
                ),
            )

        # Draw nodes
        self.node_patches = {}
        self.node_labels = {}
        self.node_status_texts = {}  # checkmark or X
        node_size = min(0.06, 0.5 / max(self.n_steps, 1))

        for node in range(self.n_steps):
            x, y = self.node_pos_norm[node]
            crit = self.task_def.steps[node].criticality

            if crit >= 2.0:
                color = C_ULTRA
            elif crit >= 1.0:
                color = C_CRITICAL
            else:
                color = C_TRIVIAL

            patch = FancyBboxPatch(
                (x - node_size, y - node_size * 0.6),
                node_size * 2, node_size * 1.2,
                boxstyle=f"round,pad={node_size * 0.15}",
                facecolor=color, edgecolor='#666666', linewidth=1.2,
                alpha=0.4,
                transform=ax.transAxes,
            )
            ax.add_patch(patch)
            self.node_patches[node] = patch

            # Label
            name = step_names[node]
            short = name[:8] if len(name) > 8 else name
            lbl = ax.text(
                x, y, f"{node}\n{short}", ha='center', va='center',
                fontsize=max(5, 8 - self.n_steps // 6),
                fontfamily='monospace',
                transform=ax.transAxes,
                alpha=0.5,
            )
            self.node_labels[node] = lbl

            # Status overlay (checkmark/X)
            status = ax.text(
                x + node_size * 0.7, y + node_size * 0.4, '',
                ha='center', va='center', fontsize=10, fontweight='bold',
                transform=ax.transAxes,
            )
            self.node_status_texts[node] = status

        # Current step indicator
        self.current_step_highlight = FancyBboxPatch(
            (0, 0), 0, 0,
            boxstyle=f"round,pad={node_size * 0.2}",
            facecolor='none', edgecolor='black', linewidth=3,
            transform=ax.transAxes, visible=False,
        )
        ax.add_patch(self.current_step_highlight)

        # Legend
        legend_items = [
            (C_TRIVIAL, 'Trivial'), (C_CRITICAL, 'Critical'), (C_ULTRA, 'Ultra-Critical')
        ]
        for i, (c, label) in enumerate(legend_items):
            ax.add_patch(plt.Rectangle(
                (0.05, 0.05 - i * 0.05), 0.04, 0.03,
                facecolor=c, edgecolor='#666', transform=ax.transAxes,
                linewidth=0.8,
            ))
            ax.text(0.10, 0.065 - i * 0.05, label, fontsize=7,
                    transform=ax.transAxes, va='center')

    def _update_graph(self, tick_idx: int):
        rec = self.records[tick_idx]
        node_size = min(0.06, 0.5 / max(self.n_steps, 1))

        # Track completed steps and failures up to this tick
        completed = set()
        failed = set()
        for r in self.records[:tick_idx + 1]:
            if r.step_completed:
                step_pos = r.true_step - 1 if r.true_step > 0 else 0
                # The step that completed is the one before current
                completed_identity = int(r.episode_order[step_pos]) if step_pos < self.n_steps else -1
                if completed_identity >= 0:
                    completed.add(completed_identity)
                if r.failure and completed_identity >= 0:
                    failed.add(completed_identity)

        for node in range(self.n_steps):
            patch = self.node_patches[node]
            lbl = self.node_labels[node]
            status = self.node_status_texts[node]

            if node in completed:
                patch.set_alpha(0.9)
                lbl.set_alpha(1.0)
                if node in failed:
                    status.set_text('X')
                    status.set_color(C_FAILED)
                    patch.set_edgecolor(C_FAILED)
                    patch.set_linewidth(2)
                else:
                    status.set_text('\u2713')
                    status.set_color(C_COMPLETED)
                    patch.set_edgecolor(C_COMPLETED)
                    patch.set_linewidth(2)
            else:
                patch.set_alpha(0.4)
                lbl.set_alpha(0.5)
                status.set_text('')
                patch.set_edgecolor('#666666')
                patch.set_linewidth(1.2)

        # Highlight current step
        current_id = rec.true_identity
        if current_id < self.n_steps and current_id in self.node_pos_norm:
            x, y = self.node_pos_norm[current_id]
            self.current_step_highlight.set_bounds(
                x - node_size * 1.1, y - node_size * 0.7,
                node_size * 2.2, node_size * 1.4,
            )
            self.current_step_highlight.set_visible(True)
            self.node_patches[current_id].set_alpha(1.0)
            self.node_labels[current_id].set_alpha(1.0)
        else:
            self.current_step_highlight.set_visible(False)

    # ------------------------------------------------------------------
    # Panel B: Action Timeline
    # ------------------------------------------------------------------
    def _init_timeline(self):
        ax = self.ax_timeline
        ax.set_title("Action Timeline", fontsize=11, fontweight='bold', pad=8)
        total_ticks = len(self.records)

        # Layout: y=0..N for step progression, two swim lanes at bottom
        ax.set_xlim(-1, total_ticks + 1)
        ax.set_ylim(-3.5, self.n_steps + 1)
        ax.set_xlabel("Tick", fontsize=9)

        # Swim lane backgrounds
        ax.axhspan(-3.5, -1.8, color='#e8f0fe', alpha=0.5)  # assistant
        ax.axhspan(-1.5, 0.2, color='#fef3e8', alpha=0.5)   # human

        ax.text(-0.5, -0.65, 'Human', fontsize=7, fontweight='bold',
                va='center', ha='right', color='#555')
        ax.text(-0.5, -2.65, 'Asst.', fontsize=7, fontweight='bold',
                va='center', ha='right', color='#555')

        # Horizontal separator
        ax.axhline(y=-1.65, color='#aaa', linewidth=0.5, linestyle='-')
        ax.axhline(y=0.3, color='#aaa', linewidth=0.5, linestyle='-')

        # Step grid
        for i in range(self.n_steps + 1):
            ax.axhline(y=i, color='#eee', linewidth=0.5)

        # Step name labels on right
        step_names = self.task_def.step_names
        for i in range(self.n_steps):
            crit = self.task_def.steps[i].criticality
            color = C_ULTRA if crit >= 2.0 else (C_CRITICAL if crit >= 1.0 else '#999')
            ax.text(total_ticks + 0.5, i + 0.5,
                    step_names[i][:12], fontsize=6, va='center',
                    color=color, fontfamily='monospace')

        ax.set_yticks(range(self.n_steps + 1))
        ax.set_yticklabels([''] * (self.n_steps + 1))

        # Pre-draw staircase line and belief line (will grow)
        self.stair_line, = ax.plot([], [], 'k-', linewidth=2, alpha=0.8,
                                    label='True step')
        self.belief_line, = ax.plot([], [], '--', color='#999', linewidth=1,
                                     alpha=0.6, label='Belief argmax')

        # Vertical progress marker
        self.progress_vline = ax.axvline(x=0, color='#ccc', linewidth=0.8,
                                          linestyle=':')

        # Action markers storage
        self.timeline_markers = []

        # Legend
        legend_handles = [
            plt.Line2D([0], [0], color=C_NARRATE, marker='|', markersize=8,
                       linewidth=0, label='Narrate'),
            plt.Line2D([0], [0], color=C_QUESTION, marker='^', markersize=6,
                       linewidth=0, label='Question'),
            plt.Line2D([0], [0], color=C_REMIND, marker='v', markersize=6,
                       linewidth=0, label='Remind'),
            plt.Line2D([0], [0], color=C_CONFIRM, marker='D', markersize=5,
                       linewidth=0, label='Confirm'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=7,
                  ncol=4, framealpha=0.8, borderpad=0.3, handletextpad=0.3,
                  columnspacing=0.8)

    def _update_timeline(self, tick_idx: int):
        ax = self.ax_timeline

        # Update staircase (true step progression)
        ticks = list(range(tick_idx + 1))
        steps = [self.records[t].true_step for t in ticks]
        self.stair_line.set_data(ticks, steps)

        # Update belief argmax
        beliefs = [np.argmax(self.records[t].step_belief) for t in ticks]
        self.belief_line.set_data(ticks, beliefs)

        # Progress line
        self.progress_vline.set_xdata([tick_idx])

        # Add new action markers (only for current tick)
        rec = self.records[tick_idx]

        # Human action
        h_name = self.h_action_names.get(rec.h_action, 'silent')
        if h_name == 'narrate':
            ax.plot(rec.tick, -0.65, '|', color=C_NARRATE, markersize=12,
                    markeredgewidth=2.5)
        elif rec.h_action in self._question_id_to_step:
            step_idx = self._question_id_to_step[rec.h_action]
            ax.plot(rec.tick, -0.65, '^', color=C_QUESTION, markersize=5)
            # Also mark which step on the staircase
            ax.plot(rec.tick, step_idx + 0.5, '^', color=C_QUESTION,
                    markersize=4, alpha=0.6)

        # Assistant action
        a_name = self.a_action_names.get(rec.a_action, 'silent')
        if a_name == 'confirm':
            ax.plot(rec.tick, -2.65, 'D', color=C_CONFIRM, markersize=4)
        elif rec.a_action in self._remind_id_to_step:
            step_idx = self._remind_id_to_step[rec.a_action]
            ax.plot(rec.tick, -2.65, 'v', color=C_REMIND, markersize=5)
            ax.plot(rec.tick, step_idx + 0.5, 'v', color=C_REMIND,
                    markersize=4, alpha=0.6)

        # Failure marker on staircase
        if rec.failure:
            ax.plot(rec.tick, rec.true_step, 'x', color=C_FAILED,
                    markersize=10, markeredgewidth=2.5)

    # ------------------------------------------------------------------
    # Panel C: Step Belief
    # ------------------------------------------------------------------
    def _init_belief(self):
        ax = self.ax_belief
        ax.set_title("Assistant's Step Belief", fontsize=9, fontweight='bold')
        n = self.n_steps + 1
        x = range(n)
        self.belief_bars = ax.bar(
            x, np.zeros(n), color=C_NARRATE, alpha=0.7, edgecolor='white',
            linewidth=0.5,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel("P", fontsize=8)
        labels = [str(i) for i in range(self.n_steps)] + ['done']
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=5, rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Argmax indicator
        self.belief_argmax_text = ax.text(
            0, 0.95, '', fontsize=8, fontweight='bold',
            transform=ax.transAxes, color=C_NARRATE,
        )

    def _update_belief(self, tick_idx: int):
        belief = self.records[tick_idx].step_belief
        for i, bar in enumerate(self.belief_bars):
            bar.set_height(belief[i] if i < len(belief) else 0)
        argmax = np.argmax(belief)
        label = str(argmax) if argmax < self.n_steps else 'done'
        self.belief_argmax_text.set_text(f"argmax={label}")

    # ------------------------------------------------------------------
    # Panel D: Memory Levels
    # ------------------------------------------------------------------
    def _init_memory(self):
        ax = self.ax_memory
        ax.set_title("Critical Step Memory", fontsize=9, fontweight='bold')

        n_crit = len(self.critical_steps)
        step_names = self.task_def.step_names
        self.crit_labels = [step_names[s][:12] for s in self.critical_steps]

        y_pos = range(n_crit)
        self.memory_bars = ax.barh(
            y_pos, np.zeros(n_crit), height=0.6,
            color=C_COMPLETED, edgecolor='white', linewidth=0.5,
        )
        ax.set_xlim(0, 2.1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.crit_labels, fontsize=7, fontfamily='monospace')
        ax.set_xlabel("Memory", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Danger threshold
        f0 = self.params.f0_base
        k = self.params.k_memory
        if k > 0 and f0 > 0:
            # f(m) = f0 * exp(-k*m) = 0.3 → m = -ln(0.3/f0)/k
            threshold = -np.log(0.3 / f0) / k if 0.3 / f0 < 1 else 0
            if 0 < threshold < 2.0:
                ax.axvline(x=threshold, color=C_FAILED, linewidth=1,
                           linestyle='--', alpha=0.6, label='30% fail')

    def _update_memory(self, tick_idx: int):
        memory = self.records[tick_idx].memory
        f0 = self.params.f0_base
        k = self.params.k_memory

        for i, step_idx in enumerate(self.critical_steps):
            m = memory[step_idx] if step_idx < len(memory) else 0
            self.memory_bars[i].set_width(m)
            # Color by failure probability
            fail_prob = f0 * np.exp(-k * m) if k > 0 else 0
            if fail_prob > 0.5:
                self.memory_bars[i].set_color(C_FAILED)
            elif fail_prob > 0.2:
                self.memory_bars[i].set_color(C_REMIND)
            else:
                self.memory_bars[i].set_color(C_COMPLETED)

    # ------------------------------------------------------------------
    # Panel E: Obs Noise + Reward
    # ------------------------------------------------------------------
    def _init_gauges(self):
        ax = self.ax_gauges
        ax.axis('off')

        # Noise display
        self.noise_text = ax.text(
            0.05, 0.8, 'Obs Noise: 0.500', fontsize=11,
            fontweight='bold', fontfamily='monospace',
            transform=ax.transAxes,
        )
        # Noise bar
        self.noise_bar_bg = ax.barh(
            [0.65], [1.0], height=0.08, left=0.05,
            color='#eee', transform=ax.transAxes,
        )
        self.noise_bar = ax.barh(
            [0.65], [0.5], height=0.08, left=0.05,
            color=C_REMIND, transform=ax.transAxes,
        )

        # Reward display
        self.reward_text = ax.text(
            0.05, 0.35, 'Reward: 0.00', fontsize=13,
            fontweight='bold', fontfamily='monospace',
            transform=ax.transAxes,
        )

        # Counters
        self.counters_text = ax.text(
            0.05, 0.08, 'Narrations: 0  Questions: 0  Failures: 0',
            fontsize=8, fontfamily='monospace',
            transform=ax.transAxes, color='#666',
        )

    def _update_gauges(self, tick_idx: int):
        rec = self.records[tick_idx]
        noise = rec.obs_noise_state
        self.noise_text.set_text(f'Obs Noise: {noise:.3f}')
        self.noise_bar[0].set_width(noise * 0.9)
        if noise < 0.1:
            self.noise_bar[0].set_color(C_COMPLETED)
        elif noise < 0.3:
            self.noise_bar[0].set_color(C_REMIND)
        else:
            self.noise_bar[0].set_color(C_FAILED)

        reward = rec.cumulative_reward
        self.reward_text.set_text(f'Reward: {reward:+.1f}')
        self.reward_text.set_color(C_COMPLETED if reward >= 0 else C_FAILED)

        # Count actions up to this tick
        n_narr = sum(1 for r in self.records[:tick_idx+1]
                     if self.h_action_names.get(r.h_action) == 'narrate')
        n_ques = sum(1 for r in self.records[:tick_idx+1]
                     if r.h_action in self._question_id_to_step)
        n_fail = sum(1 for r in self.records[:tick_idx+1] if r.failure)
        self.counters_text.set_text(
            f'Narrations: {n_narr}  Questions: {n_ques}  Failures: {n_fail}'
        )

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _init_artists(self):
        """Initialize all panels."""
        self._init_graph()
        self._init_timeline()
        self._init_belief()
        self._init_memory()
        self._init_gauges()
        return []

    def _update(self, frame_idx):
        """Update all panels for given frame."""
        tick_idx = self.frame_to_tick[frame_idx]

        # Update header
        rec = self.records[tick_idx]
        probe_str = f"Probe {self.probe_id}" if self.probe_id else "Standard"
        step_name = (self.task_def.step_names[rec.true_identity]
                     if rec.true_identity < self.n_steps else "DONE")
        self.title_text.set_text(
            f"Task: {self.task_name} ({self.n_steps} steps)  |  "
            f"Regime: {self.regime_name}  |  {probe_str}  |  "
            f"Tick: {rec.tick} / {len(self.records)}  |  "
            f"Step: {step_name}"
        )

        self._update_graph(tick_idx)
        # For timeline, we need to draw all ticks up to current
        # But only add new markers for current tick
        self._update_timeline(tick_idx)
        self._update_belief(tick_idx)
        self._update_memory(tick_idx)
        self._update_gauges(tick_idx)

        return []

    def save(self, output_path: Path, fps: int = 4):
        """Save animation as MP4 (with GIF fallback)."""
        self._setup_figure()

        # Init all panels once
        self._init_artists()

        anim = FuncAnimation(
            self.fig, self._update,
            frames=self.total_frames,
            interval=1000 // fps,
            blit=False,
            repeat=False,
        )

        # Try ffmpeg first (via imageio_ffmpeg)
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=3000)
            mp4_path = output_path.with_suffix('.mp4')
            anim.save(str(mp4_path), writer=writer, dpi=100)
            print(f"MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"FFMpeg failed ({e}), falling back to GIF...")
            from matplotlib.animation import PillowWriter
            gif_path = output_path.with_suffix('.gif')
            writer = PillowWriter(fps=fps)
            anim.save(str(gif_path), writer=writer, dpi=80)
            print(f"GIF saved: {gif_path}")

        plt.close(self.fig)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Animate a multi-agent episode with task graph visualization')
    parser.add_argument('--task', default='make_cereal',
                        help='Task name (default: make_cereal)')
    parser.add_argument('--regime', default='extremely_high',
                        choices=['extremely_low', 'balanced', 'extremely_high'])
    parser.add_argument('--probe', default=None, choices=['A', 'B'],
                        help='Narration probe ID (default: standard)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--output', default=None,
                        help='Output path (default: auto-generated)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (default: stochastic)')
    args = parser.parse_args()

    print(f"Recording episode: {args.task} / {args.regime}"
          f" / {'Probe ' + args.probe if args.probe else 'standard'}"
          f" / {'deterministic' if args.deterministic else 'stochastic'}")

    # Record episode
    recorder = EpisodeRecorder()
    records, ma_env, params = recorder.run_episode(
        args.task, args.regime, args.probe, args.seed, args.deterministic,
    )

    print(f"Episode: {len(records)} ticks, "
          f"reward={records[-1].cumulative_reward:.2f}, "
          f"narrations={sum(1 for r in records if ma_env.get_action_names()[0].get(r.h_action) == 'narrate')}, "
          f"questions={sum(1 for r in records if r.h_action in ma_env._question_id_to_step)}")

    # Build output path
    if args.output:
        output_path = Path(args.output)
    else:
        probe_str = f"_probe{args.probe}" if args.probe else ""
        output_path = (PROJECT_ROOT / "results" / "videos" /
                       f"ma_episode_{args.task}_{args.regime}{probe_str}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Animate
    animator = MAEpisodeAnimator(
        records, ma_env, params,
        args.task, args.regime, args.probe,
    )
    animator.save(output_path, fps=args.fps)


if __name__ == '__main__':
    main()
