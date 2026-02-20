"""
Cross-Task Multi-Regime Report Generator

Creates markdown report with comprehensive comparison table.

Usage:
    python generate_cross_task_report.py
"""

import json
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent


def generate_report():
    """Generate markdown report from evaluation results."""
    results_path = PROJECT_ROOT / "data" / "results" / "cross_task_evaluation_results.json"

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run evaluate_cross_task_all_regimes.py first!")
        return

    # Load results
    df = pd.read_json(results_path)

    # Filter successful evaluations
    df_success = df[df['status'] == 'success']

    if len(df_success) == 0:
        print("Error: No successful evaluations found!")
        return

    # Create markdown table
    markdown = f"""# Cross-Task Multi-Regime Experiment Results

## Configuration

- **Tasks**: 7 (make_cereal, make_coffee, make_tea, make_sandwich, make_stencil, cooking, latte_making)
- **Regimes**: 3 (very_high_stakes, balanced, moderate_low)
- **Total Models**: 21
- **Training**: 200,000 timesteps per model
- **Baseline Failure Rate**: 60% (f0_base=0.6)
- **Action Space**: Sparse (only 2-4 critical steps per task)

## Summary Table

| Task | Regime | RL Reward | RL Interventions | RL Failures | Random Reward | Improvement |
|------|--------|-----------|------------------|-------------|---------------|-------------|
"""

    # Add rows
    for _, row in df_success.iterrows():
        markdown += f"| {row['task']} | {row['regime']} | "
        markdown += f"{row['rl_reward_mean']:.1f} ± {row['rl_reward_std']:.1f} | "
        markdown += f"{row['rl_interventions_mean']:.2f} | "
        markdown += f"{row['rl_failures_mean']:.2f} | "
        markdown += f"{row['random_reward_mean']:.1f} | "
        markdown += f"{row['improvement_pct']:.1f}% |\n"

    # Add summary statistics
    best_idx = df_success['rl_reward_mean'].idxmax()
    best_improvement_idx = df_success['improvement_pct'].idxmax()

    markdown += f"""

## Key Findings

- **Total Successful Evaluations**: {len(df_success)}/21
- **Best Performing Model**: {df_success.loc[best_idx, 'task']} / {df_success.loc[best_idx, 'regime']} (reward: {df_success.loc[best_idx, 'rl_reward_mean']:.1f})
- **Largest Improvement**: {df_success.loc[best_improvement_idx, 'task']} / {df_success.loc[best_improvement_idx, 'regime']} (+{df_success.loc[best_improvement_idx, 'improvement_pct']:.1f}%)
- **Overall Mean Improvement**: {df_success['improvement_pct'].mean():.1f}%
- **Median Improvement**: {df_success['improvement_pct'].median():.1f}%

## Intervention Analysis

### Mean Interventions per Episode by Regime

"""

    # Group by regime
    for regime in ['very_high_stakes', 'balanced', 'moderate_low']:
        regime_df = df_success[df_success['regime'] == regime]
        if len(regime_df) > 0:
            markdown += f"- **{regime}**: {regime_df['rl_interventions_mean'].mean():.2f} (avg across {len(regime_df)} tasks)\n"

    markdown += f"""

## Failure Analysis

### Mean Failures per Episode by Regime

"""

    for regime in ['very_high_stakes', 'balanced', 'moderate_low']:
        regime_df = df_success[df_success['regime'] == regime]
        if len(regime_df) > 0:
            markdown += f"- **{regime}**: {regime_df['rl_failures_mean'].mean():.2f} (avg across {len(regime_df)} tasks)\n"

    # Count models with non-zero interventions
    with_interventions = len(df_success[df_success['rl_interventions_mean'] > 0])
    pct_with_interventions = (with_interventions / len(df_success)) * 100

    markdown += f"""

## Success Criteria Check

- **Models with non-zero interventions**: {with_interventions}/{len(df_success)} ({pct_with_interventions:.1f}%)
- **Models with >20% improvement**: {len(df_success[df_success['improvement_pct'] > 20])}/{len(df_success)}
- **Models with <2 failures on average**: {len(df_success[df_success['rl_failures_mean'] < 2.0])}/{len(df_success)}

## Conclusion

{
"✓ SUCCESS: Agent discovered strategic reminding!" if with_interventions > len(df_success) * 0.5
else "⚠ PARTIAL: Some models learned reminding, but not majority."
}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    output_path = PROJECT_ROOT / "results" / "cross_task_experiment_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"\n{'='*80}")
    print(f"REPORT GENERATED")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")
    print(f"  Models evaluated: {len(df_success)}/21")
    print(f"  Mean improvement: {df_success['improvement_pct'].mean():.1f}%")
    print(f"{'='*80}\n")

    # Print report to console as well
    print(markdown)


if __name__ == "__main__":
    generate_report()
