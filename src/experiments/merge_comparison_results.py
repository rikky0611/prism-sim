"""
Merge multiple per-task comparison_3policy JSON files into one.

Usage:
    python merge_comparison_results.py \
        data/results/comparison_3policy.json \
        data/results/comparison_3policy_make_coffee.json \
        data/results/comparison_3policy_make_tea.json \
        -o data/results/comparison_3policy_all.json
"""

import sys
import json
import argparse
from pathlib import Path


def merge_results(input_paths: list[Path]) -> dict:
    """Merge multiple comparison result JSONs by combining their conditions dicts.

    Each input file is expected to have a top-level 'conditions' dict keyed by
    task name. The merged output unions all task keys.

    Args:
        input_paths: List of JSON file paths to merge.

    Returns:
        Merged result dictionary.
    """
    merged = None

    for path in input_paths:
        with open(path) as f:
            data = json.load(f)

        if merged is None:
            merged = data.copy()
            merged['source_files'] = [str(path)]
            continue

        merged['source_files'].append(str(path))

        # Merge conditions (task-level keys)
        for task_name, task_data in data.get('conditions', {}).items():
            if task_name in merged['conditions']:
                # Deep-merge obs_noise -> fail_regime level
                for obs_key, obs_data in task_data.items():
                    merged['conditions'][task_name].setdefault(obs_key, {})
                    merged['conditions'][task_name][obs_key].update(obs_data)
            else:
                merged['conditions'][task_name] = task_data

    if merged is None:
        merged = {'conditions': {}}

    # Update metadata
    all_tasks = sorted(merged['conditions'].keys())
    merged['tasks_merged'] = all_tasks
    merged['n_tasks'] = len(all_tasks)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-task comparison_3policy JSON files'
    )
    parser.add_argument(
        'inputs', nargs='+',
        help='Input JSON files to merge'
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Output merged JSON path'
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    merged = merge_results(input_paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    tasks = merged.get('tasks_merged', [])
    n_conditions = sum(
        len(fail_data)
        for task_data in merged['conditions'].values()
        for obs_data in task_data.values()
        for fail_data in [obs_data]
    )
    print(f"Merged {len(input_paths)} files -> {output_path}")
    print(f"  Tasks: {tasks}")
    print(f"  Total condition groups: {n_conditions}")


if __name__ == '__main__':
    main()
