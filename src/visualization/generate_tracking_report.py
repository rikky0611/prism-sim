"""
Generate a markdown report summarizing the semi-Markov belief sweep results.

Reads data/results/comparison_v3_semi_markov_all_tasks.json and the figures
produced by plot_tracking_sweep.py, then assembles a markdown report with
inline tables and figure references at docs/english/TRACKING_SWEEP_REPORT.md.

Usage:
    python3 generate_tracking_report.py
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
NOISE_ORDER = ['low', 'normal', 'high']
FAIL_ORDER = ['low', 'balanced', 'high']
POLICIES = ['passive', 'heuristic', 'ma_ippo']
POLICY_LABELS = {'passive': 'Passive', 'heuristic': 'Heuristic', 'ma_ippo': 'MA-IPPO'}


def _load(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _cell(results, task, noise, fail, policy, metric):
    cell = results['conditions'].get(task, {}).get(noise, {}).get(fail, {})
    pol = cell.get(policy)
    if pol is None:
        return None
    v = pol.get(metric)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _task_tracking_table(results, task) -> str:
    rows = ['| noise \\\\ fail | low | balanced | high |',
            '|---|---|---|---|']
    for noise in NOISE_ORDER:
        cells = [f'**{noise}**']
        for fail in FAIL_ORDER:
            parts = []
            for policy in POLICIES:
                v = _cell(results, task, noise, fail, policy, 'mean_tracking_map_acc')
                if v is None:
                    parts.append(f'{POLICY_LABELS[policy][0]}: n/a')
                else:
                    parts.append(f'{POLICY_LABELS[policy][0]}: {v:.2f}')
            cells.append('<br>'.join(parts))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def _cross_task_mean_table(results, tasks, metric, policy) -> str:
    rows = ['| noise \\\\ fail | low | balanced | high |',
            '|---|---|---|---|']
    for noise in NOISE_ORDER:
        cells = [f'**{noise}**']
        for fail in FAIL_ORDER:
            vals = [_cell(results, t, noise, fail, policy, metric) for t in tasks]
            vals = [v for v in vals if v is not None]
            if not vals:
                cells.append('n/a')
            else:
                m = np.mean(vals)
                s = np.std(vals)
                cells.append(f'{m:.2f} ± {s:.2f}')
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def _policy_tracking_summary(results, tasks, policy) -> Dict[str, float]:
    """Mean tracking accuracy per obs_noise across all fails × tasks."""
    out = {}
    for noise in NOISE_ORDER:
        vals = []
        for t in tasks:
            for f in FAIL_ORDER:
                v = _cell(results, t, noise, f, policy, 'mean_tracking_map_acc')
                if v is not None:
                    vals.append(v)
        out[noise] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(
        PROJECT_ROOT / 'data' / 'results' /
        'comparison_v3_semi_markov_all_tasks.json'))
    ap.add_argument('--figures-dir', default='results/figures/tracking_sweep',
                    help='Path to figures dir, relative to docs location')
    ap.add_argument('--out', default=str(
        PROJECT_ROOT / 'docs' / 'english' / 'TRACKING_SWEEP_REPORT.md'))
    args = ap.parse_args()

    results = _load(Path(args.results))
    preferred = ['make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
                 'cooking', 'make_stencil', 'latte_making']
    tasks = [t for t in preferred if t in results['conditions']]

    n_ep = results.get('n_eval_episodes', 'n/a')
    n_tasks = len(tasks)

    # Figure paths — use paths that work when report is viewed from repo root
    fig = lambda name: f'../../{args.figures_dir}/{name}'

    # Cross-task mean tracking per policy
    summary = {p: _policy_tracking_summary(results, tasks, p) for p in POLICIES}

    lines = []
    lines.append('# Semi-Markov Belief Sweep — Tracking Accuracy Across Tasks')
    lines.append('')
    lines.append(f'**Tasks ({n_tasks}):** `' + '`, `'.join(tasks) + '`')
    lines.append(f'**Conditions:** obs_noise ∈ {{low, normal, high}} × fail_regime ∈ {{low, balanced, high}}')
    lines.append(f'**Policies:** Passive · Heuristic · MA-IPPO (IPPO trained per condition)')
    lines.append(f'**Evaluation:** {n_ep} episodes per condition.')
    lines.append('')
    lines.append('## 1. What changed')
    lines.append('')
    lines.append('The assistant now maintains a **PrISM-style semi-Markov joint belief** '
                 '`b(s, τ)` over (step identity, elapsed dwell τ) instead of the previous '
                 'step-only marginal. Belief propagation uses a per-step Gaussian dwell '
                 'model `N(μ_s, σ_s)` with escape probability '
                 '`1 − Φ̄(τ+1)/Φ̄(τ)`, a transition matrix built from the task\'s rollout '
                 'patterns, and Bayesian updates against noisy identity observations.')
    lines.append('')
    lines.append('Evaluation now reports **`tracking_map_acc`**: the per-tick fraction '
                 'of ticks where `argmax(belief_marginal) == true_identity`, averaged across '
                 'the episode and then across episodes. Tick 0 and post-done ticks are '
                 'excluded so the metric captures non-trivial belief quality.')
    lines.append('')
    lines.append('## 2. Main tracking-accuracy result')
    lines.append('')
    lines.append('Per-policy tracking accuracy heatmaps over the obs_noise × fail_regime '
                 'grid, averaged across tasks.')
    lines.append('')
    lines.append(f'![tracking heatmaps per policy]({fig("tracking_heatmaps_per_policy.png")})')
    lines.append('')
    lines.append('**Aggregated across fail_regime × task:**')
    lines.append('')
    lines.append('| Policy | Low noise | Normal noise | High noise |')
    lines.append('|---|---|---|---|')
    for p in POLICIES:
        row = [POLICY_LABELS[p]]
        for n in NOISE_ORDER:
            m, s = summary[p][n]
            row.append(f'{m:.3f} ± {s:.3f}')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('**Observations**')
    lines.append('')
    lines.append('- Tracking accuracy for `Passive` and `Heuristic` is essentially the same '
                 'in every cell (within noise). This is the expected invariant: the belief '
                 'is maintained environment-side and updated from the human\'s actions and '
                 'sensor observations — the assistant\'s reminder policy has no causal path '
                 'to the belief.')
    lines.append('- Accuracy degrades monotonically from low → high observation noise, as '
                 'the Bayesian filter has weaker evidence to concentrate its posterior.')
    lines.append('- `MA-IPPO` outperforms the passive baseline on tracking only when the '
                 'learned human narrates. Narration resets the belief to a point mass on '
                 'the true identity, and the learned controller turns narration on/off '
                 'based on the cost trade-off.')
    lines.append('')
    lines.append('## 3. Narration explains the MA-IPPO tracking gap')
    lines.append('')
    lines.append(f'![narration heatmap]({fig("narration_heatmap_ma_ippo.png")})')
    lines.append('')
    lines.append(f'![tracking vs narration]({fig("tracking_vs_narration.png")})')
    lines.append('')
    lines.append('The learned human\'s narration frequency tracks the tracking accuracy '
                 'MA-IPPO achieves. Where narration ≈ 0 (low fail cost, narration doesn\'t '
                 'pay off), MA-IPPO\'s tracking collapses to the passive baseline; where '
                 'narration ramps up (high fail cost + high noise), tracking approaches '
                 '1.0. This is the expected semi-Markov behaviour: narration is a perfect '
                 'observation in the filter, so each narration pins the belief.')
    lines.append('')
    lines.append('## 4. Per-task tracking accuracy (Passive)')
    lines.append('')
    lines.append('Per-task view of the policy-independent tracking accuracy (Passive). '
                 'The pattern is consistent across tasks: accuracy falls with noise and '
                 'is roughly invariant to fail_regime (fail cost changes the reward '
                 'landscape but not the sensor/belief dynamics).')
    lines.append('')
    lines.append(f'![tracking per task]({fig("tracking_heatmaps_per_task.png")})')
    lines.append('')
    lines.append('### Passive tracking accuracy per (task, noise, fail)')
    lines.append('')
    lines.append(_cross_task_mean_table(results, tasks, 'mean_tracking_map_acc', 'passive'))
    lines.append('')
    lines.append('## 5. Reward comparison across tasks')
    lines.append('')
    lines.append(f'![reward bars]({fig("reward_bars_by_condition.png")})')
    lines.append('')
    lines.append('Bars are mean reward across the 7 tasks (higher is better).')
    lines.append('')
    lines.append('**Per-condition mean reward, averaged across tasks:**')
    lines.append('')
    lines.append('| Policy | Low/Low | Low/Bal | Low/High | N/Low | N/Bal | N/High | H/Low | H/Bal | H/High |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for p in POLICIES:
        row = [POLICY_LABELS[p]]
        for n in NOISE_ORDER:
            for f in FAIL_ORDER:
                vals = [_cell(results, t, n, f, p, 'mean_reward') for t in tasks]
                vals = [v for v in vals if v is not None]
                row.append(f'{np.mean(vals):+.1f}' if vals else 'n/a')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    lines.append('## 6. Tracking accuracy by observation noise (grouped bars)')
    lines.append('')
    lines.append(f'![tracking bars]({fig("tracking_bars_by_noise.png")})')
    lines.append('')
    lines.append('## 7. Per-task quick tables')
    lines.append('')
    lines.append('Each cell shows `P`assive / `H`euristic / `M`A-IPPO tracking accuracy.')
    lines.append('')
    for task in tasks:
        lines.append(f'### {task}')
        lines.append('')
        lines.append(_task_tracking_table(results, task))
        lines.append('')
    lines.append('## 8. Caveats and scope')
    lines.append('')
    lines.append('- Training budget was kept small (3 rounds × 15k steps per condition '
                 'for newly-trained models; make_cereal used the earlier 5 × 20k budget). '
                 'The learned policies are directionally meaningful but not fully '
                 'converged; absolute rewards should be treated as a lower bound on '
                 'what longer training can achieve.')
    lines.append('- 100 evaluation episodes per condition. Means are stable within '
                 '~0.02 std, which is enough to read the tracking-vs-noise trend cleanly '
                 'but may smear fine differences between Passive and Heuristic.')
    lines.append('- Observation noise is treated as a generative parameter only; the '
                 'tracker\'s Gaussian-on-integer likelihood model is an approximation '
                 'matching PrISM-Tracker\'s confusion-matrix-calibrated HAR outputs.')
    lines.append('- The transition graph used for belief prior propagation is built from '
                 'the task\'s `rollout_patterns` and equals the generating distribution, '
                 'i.e., the belief prior is effectively the ground-truth graph. Separating '
                 'the tracker\'s graph from the generator (to study '
                 'misspecification) is deferred.')
    lines.append('')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
