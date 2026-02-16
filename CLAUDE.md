# Claude Code Instructions - Procedure Assistant Research Project

## Project Overview

**Research Topic**: Learning Context-Dependent Procedure Assistant Policies with Deep Reinforcement Learning

This project explores using Deep RL (PPO - Proximal Policy Optimization) to learn adaptive policies for AI procedure assistants that optimize the tradeoff between interruption costs and failure costs in procedural tasks. The research demonstrates that RL agents can discover context-dependent strategies ranging from "strategic silence" to active intervention based on cost structures.

**Key Technologies**:
- Python 3.9+
- Stable-Baselines3 (PPO implementation)
- Gymnasium (RL environment)
- Matplotlib/Seaborn (visualization)
- python-pptx (presentations)

**Current Status**: Core experiments complete, multi-regime analysis complete, presentation generated

---

## Directory Structure

```
/Antti/
├── src/                          # Source code (DO NOT output files here)
│   ├── training/                 # RL training scripts
│   ├── experiments/              # Experiment runners
│   └── visualization/            # Plotting and presentation scripts
│
├── data/                         # Data files
│   ├── results/                  # JSON result files (output here)
│   └── logs/                     # Training logs (output here)
│
├── results/                      # Outputs and deliverables
│   ├── figures/                  # PNG visualizations (output here)
│   ├── presentations/            # PowerPoint files (output here)
│   └── videos/                   # MP4 demonstrations
│
├── models/                       # Trained model checkpoints (output here)
│   ├── balanced/
│   ├── multi_regime_v1/
│   ├── multi_regime_v2/
│   └── tensorboard/
│
├── docs/                         # Documentation
│   ├── english/                  # English documentation
│   ├── japanese/                 # Japanese documentation
│   └── papers/                   # Academic papers
│
├── CLAUDE.md                     # This file
├── README.md                     # Project README
├── .gitignore                    # Git ignore rules
└── venv/                         # Python virtual environment
```

---

## 🚨 FILE ORGANIZATION POLICY (CRITICAL)

### **RULE 1: Always organize files immediately after generation**

When you create or generate ANY file, you MUST immediately move it to the appropriate directory:

| File Type | Destination | Examples |
|-----------|-------------|----------|
| Python scripts | `src/{training,experiments,visualization}/` | `train_*.py`, `run_*.py`, `visualize_*.py`, `generate_*.py` |
| JSON results | `data/results/` | `*_results.json`, `experiment_*.json` |
| Training logs | `data/logs/` | `*_log.txt`, `*_output.txt` |
| PNG/JPG images | `results/figures/` | `*.png`, `*.jpg`, `*_comparison.png` |
| PowerPoint | `results/presentations/` | `*.pptx` |
| Videos | `results/videos/` | `*.mp4` |
| Model checkpoints | `models/{regime_name}/` | `ppo_assistant_*/` directories |
| Markdown docs | `docs/english/` or `docs/japanese/` | `*.md` (except README, CLAUDE) |
| PDFs | `docs/papers/` | `*.pdf` |

### **RULE 2: Never leave files at project root**

The project root should ONLY contain:
- `CLAUDE.md` (this file)
- `README.md`
- `.gitignore`
- Configuration directories (`.vscode/`, `.claude/`, `__pycache__/`)
- Virtual environment (`venv/`)

### **RULE 3: Update paths when moving files**

After moving files, ALWAYS update:
1. Import statements in Python scripts
2. File paths in scripts that read/write data
3. References in documentation
4. README.md examples if affected

### **RULE 4: Verify scripts still work**

After reorganizing, test that key scripts run:
```bash
cd src/training && python train_rl_policy.py --help
cd src/experiments && python run_experiments.py --help
```

---

## Path Management Guidelines

### For Python Scripts in `src/`

Use **relative paths from the script's location**:

```python
# Example: src/training/train_rl_policy.py
import os
from pathlib import Path

# Get project root (2 levels up from script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Save model to models/
model_path = PROJECT_ROOT / "models" / "balanced" / "ppo_assistant_balanced"
model.save(str(model_path / "final_model"))

# Save results to data/results/
results_path = PROJECT_ROOT / "data" / "results" / "rl_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f)

# Save figures to results/figures/
figure_path = PROJECT_ROOT / "results" / "figures" / "comparison.png"
plt.savefig(figure_path)
```

### Benefits of This Approach
- Works regardless of where script is run from
- Cross-platform compatible (Windows/Mac/Linux)
- Type-safe with pathlib
- Easy to understand and maintain

---

## Common Workflows

### Training a New Model

```bash
# From project root
cd src/training

# Train single regime
python train_rl_policy.py --regime balanced --timesteps 50000

# Train multi-regime
python train_multi_regime_v2.py

# Models will be saved to ../../models/
```

### Running Experiments

```bash
# From project root
cd src/experiments

# Run all experiments
python run_experiments.py

# Results saved to ../../data/results/
# Figures saved to ../../results/figures/
```

### Generating Visualizations

```bash
# From project root
cd src/visualization

# Generate presentation
python generate_presentation.py

# Output saved to ../../results/presentations/
```

### Viewing Documentation

```bash
# From project root
cat docs/english/README.md
cat docs/english/RL_RESULTS_ANALYSIS.md
```

---

## Code Style Guidelines

### Python Standards
- **PEP 8 compliance**: Use 4 spaces, max 100 chars per line
- **Type hints**: Add for all function signatures
  ```python
  def train_policy(params: SimulationParams, timesteps: int = 50000) -> PPO:
      ...
  ```
- **Docstrings**: Use for all modules, classes, and public functions
  ```python
  def compare_policies(model: PPO, n_episodes: int = 100) -> Dict[str, Any]:
      """Compare RL policy against baseline heuristics.

      Args:
          model: Trained PPO model
          n_episodes: Number of evaluation episodes

      Returns:
          Dictionary with policy names as keys and metrics as values
      """
  ```
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Import Organization
```python
# Standard library
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Local
from procedure_assistant_sim import ProcedureAssistantEnv, SimulationParams
```

---

## Documentation Requirements

### When to Update Documentation

**Always update docs when:**
- Adding new experiments → Update `docs/english/` with analysis
- Changing core algorithms → Update `docs/english/IMPLEMENTATION_*.md`
- Getting significant results → Create new analysis document
- Changing directory structure → Update this CLAUDE.md
- Adding features → Update README.md

### Documentation File Types

| Type | Location | Purpose |
|------|----------|---------|
| README | Root + `docs/english/` | Quick start, overview |
| Analysis docs | `docs/english/` | Experiment results, insights |
| Implementation docs | `docs/english/` | Design decisions, reflections |
| Reports | `docs/japanese/` | Japanese research reports |
| Papers | `docs/papers/` | Academic literature |

---

## Git Workflow

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance (deps, config)
- `organize`: File reorganization

**Examples:**
```
feat: implement multi-regime RL training

- Add train_multi_regime_v2.py with 5 cost regimes
- Support adaptive intervention strategies
- Increase failure risk (f0=0.6, lambda=0.10)

organize: restructure project directories

- Move Python scripts to src/
- Move results to data/ and results/
- Move models to models/ with regime subdirs
- Update all import paths and file references
```

### What to Track vs Ignore

**Track (commit to git):**
- Source code (`src/`)
- Documentation (`docs/`)
- Small result files (`data/results/*.json`)
- Key figures (`results/figures/*.png`)
- CLAUDE.md, README.md, .gitignore

**Ignore (in .gitignore):**
- Large model files (`models/*/`)
- Virtual environment (`venv/`)
- Videos (`results/videos/*.mp4`)
- Logs (`data/logs/*.txt`)
- Python cache (`__pycache__/`, `*.pyc`)
- System files (`.DS_Store`, `.vscode/`)

---

## Working with Claude Code

### Communication Style Preferences

**Do:**
- Be concise and direct
- Ask clarifying questions if requirements are unclear
- Suggest improvements proactively
- Organize files immediately after generation
- Test code before declaring it complete

**Don't:**
- Generate files without organizing them
- Make assumptions about ambiguous requirements
- Skip path updates after moving files
- Leave broken imports or references
- Create files at project root (except CLAUDE.md, README, .gitignore)

### Request Patterns

| User Request | Your Action |
|--------------|-------------|
| "Train a new model for X" | 1. Create/modify script in `src/training/`<br>2. Run training<br>3. Move output to `models/`<br>4. Document results in `docs/english/` |
| "Generate visualization of Y" | 1. Create/modify script in `src/visualization/`<br>2. Generate PNG<br>3. Move to `results/figures/`<br>4. Confirm output location |
| "Analyze experiment Z" | 1. Read data from `data/results/`<br>2. Perform analysis<br>3. Create markdown doc in `docs/english/`<br>4. Generate figures if needed → `results/figures/` |

### When to Ask for Clarification

Always ask when:
- User request could have multiple interpretations
- Unclear which regime/parameters to use
- Ambiguous where output should go (if not covered by this guide)
- Uncertain whether to overwrite existing files
- User asks for "the usual" or references prior work without specifics

---

## Post-Generation Maintenance Checklist

After ANY code generation session, verify:

- [ ] **Files organized**: All new files in correct directories
- [ ] **Paths updated**: Import statements and file paths corrected
- [ ] **Scripts tested**: Key scripts run without errors
  ```bash
  cd src/training && python train_rl_policy.py --help
  cd src/experiments && python run_experiments.py --help
  ```
- [ ] **Documentation current**: Relevant docs updated with changes
- [ ] **Root clean**: Only approved files at project root (CLAUDE.md, README.md, .gitignore, config dirs, venv)
- [ ] **Git status clean**: Unintended files not staged
  ```bash
  git status  # Review what's changed
  ```

---

## Quick Reference Card

### File Generation Decision Tree

```
Generated a file → What type?
├─ Python script
│  ├─ Training → src/training/
│  ├─ Experiment → src/experiments/
│  └─ Visualization → src/visualization/
├─ Data file
│  ├─ JSON results → data/results/
│  └─ Log file → data/logs/
├─ Output file
│  ├─ PNG/JPG → results/figures/
│  ├─ PowerPoint → results/presentations/
│  └─ Video → results/videos/
├─ Model checkpoint → models/{regime_name}/
└─ Documentation
   ├─ English → docs/english/
   ├─ Japanese → docs/japanese/
   └─ PDF → docs/papers/
```

### Path Patterns

```python
# From src/training/ or src/experiments/ or src/visualization/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Common paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
```

---

## Version History

- **2026-02-16**: Initial CLAUDE.md created with file organization policy
- **Future**: Track major structural changes here

---

**Remember**: A well-organized project is easier to navigate, maintain, and share. Always organize files immediately after generation! 🎯
