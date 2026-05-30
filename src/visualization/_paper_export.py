"""Helper for plot scripts: optionally also copy a saved figure to the
paper figure folder (`results/figures/paper/`). Used by `--paper` CLI flag.

Also exports `V5_FAILURE_TAG`: the canonical suptitle descriptor of the
v5 failure model. v5 dynamics use fixed constants
(c_fail_critical = 50 with episode termination on critical failure;
c_fail_noncritical = 1 with episode continuing). The legacy `c_fail_scale`
parameter has no effect on dynamics; plotters should use V5_FAILURE_TAG
in suptitles rather than printing the vestigial value.
"""

from pathlib import Path
from shutil import copy

PAPER_FIG_DIR = (
    Path(__file__).parent.parent.parent / 'results' / 'figures' / 'paper'
)

V5_FAILURE_TAG = 'R_fail_crit=-50 (terminal), R_fail_noncrit=-1'


def export_to_paper(output_path: str, *, paper: bool = False,
                    extra_paths: list = None) -> None:
    """If paper=True, copy `output_path` (and any in `extra_paths`) into
    `results/figures/paper/`, preserving filename. Idempotent."""
    if not paper:
        return
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [output_path] + list(extra_paths or [])
    for p in paths:
        src = Path(p)
        if not src.exists():
            continue
        dst = PAPER_FIG_DIR / src.name
        copy(src, dst)
        print(f'Also exported to paper folder: {dst}')


def add_paper_arg(parser) -> None:
    """Add a standard --paper flag to an argparse parser."""
    parser.add_argument(
        '--paper', action='store_true',
        help=f'Also copy output PNG into paper figures folder ({PAPER_FIG_DIR})',
    )
