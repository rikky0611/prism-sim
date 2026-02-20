# Explanation Diagrams for Procedure Assistant RL Experiment

This directory contains 4 publication-quality diagrams designed to explain the procedural assistance RL experiment to professors, advisors, or conference audiences.

## Diagram Overview

### 1. System Architecture (`explanation_1_system_architecture.png`)
**Purpose**: Explains the high-level system and interaction loop

**Shows**:
- **Human Agent**: Performing the procedural task (e.g., making coffee, cooking)
- **Procedure Assistant**: AI agent that observes and decides when to intervene with reminders
- **Environment**: Maintains task state (current step, memory levels, failures)
- **Information Flow**:
  - Observation (partial): Environment → Assistant
  - Action (silent/remind): Assistant → Human
  - State Update: Human actions → Environment
  - Reward Signal: Negative cost from failures and interruptions

**Key Insight**: This is a POMDP where the assistant must learn when to act based on partial observations of memory state.

**Use for**: Introducing the problem setup, explaining "what is being modeled"

---

### 2. Task Structure (`explanation_2_task_structure.png`)
**Purpose**: Shows what procedural tasks look like and how criticality varies

**Shows**:
- **Two example tasks**:
  - make_cereal: 8 steps (simple)
  - latte_making: 20 steps (complex)
- **Each step displays**:
  - Step number and name
  - Criticality value (color-coded)
  - Failure cost (base_cost × criticality)
- **Color coding**:
  - Gray: Trivial steps (criticality = 0.0, cannot fail)
  - Orange: Critical steps (criticality = 1.0, normal failure risk)
  - Red: Ultra-critical steps (criticality > 1.0, high failure risk)

**Key Insight**: Not all steps are equally important - some are trivial (getting bowl), others are critical (pouring cereal). The RL agent must learn which steps need attention.

**Use for**: Explaining the domain, showing task diversity, motivating why intervention strategy should vary by step

---

### 3. Memory Dynamics (`explanation_3_memory_dynamics.png`)
**Purpose**: Explains the core mechanism driving optimal behavior

**Shows**:
- **Top panel**: Memory evolution over time
  - Memory decay: Exponential decay (λ=0.05, 14-tick half-life)
  - Intervention at tick 30: Shows memory boost (+0.6)
  - Low memory threshold (0.3): When failure risk becomes significant

- **Bottom panel**: Resulting failure probability
  - Formula: f(m) = f₀ × exp(-k×m)
  - Shows dramatic reduction after intervention
  - Gradual increase as memory decays
  - Concrete examples showing failure rates at different memory levels:
    - m=0.0 → 60% failure rate (no memory)
    - m=0.3 → 24% failure rate (low memory)
    - m=0.6 → 10% failure rate (good memory)
    - m=1.0 → 3% failure rate (excellent memory)

**Key Insight**:
1. Reminders restore memory (+0.6 boost) but it decays exponentially
2. Timing matters - intervening too early wastes the boost before critical steps
3. Failure probability is exponentially related to memory level (20× reduction from m=0 to m=1)

**Use for**: Explaining the failure model, showing why "when to intervene" is a temporal credit assignment problem

---

### 4. Cost Structure & Strategy (`explanation_4_cost_strategy.png`)
**Purpose**: Shows why different cost regimes lead to different learned behaviors

**Shows**:
- **Left panel**: Cost regime configuration table
  | Regime | c_fail | c_int | Ratio | Expected Behavior |
  |--------|--------|-------|-------|-------------------|
  | Very High Stakes | 30 | 1 | 30:1 | Rare interventions |
  | Balanced | 15 | 1 | 15:1 | Mixed strategy |
  | Moderate Low | 10 | 1 | 10:1 | Minimal interventions |

- **Right panel**: Optimal strategy heatmap
  - X-axis: Failure probability (0 to 1)
  - Y-axis: Cost ratio (c_fail / c_int)
  - Color: Blue = stay silent, Red = intervene
  - Decision boundary: Intervene if p_fail > c_int / c_fail
  - Regime markers show where each regime operates

**Key Insight**:
1. Higher cost ratios → intervene only at very high failure probabilities
2. Lower cost ratios → intervene more liberally
3. Optimal strategy is deterministic threshold policy based on cost-benefit analysis
4. RL agents discover this threshold through learning

**Use for**: Explaining why we expect different behaviors, showing the theoretical optimal strategy, validating that RL learns sensible policies

---

## Usage Guide

### For Presentations

**Slide 1-2: Introduction**
- Use **Diagram 1 (System Architecture)** to introduce the problem
- Explain: "We're training an RL agent to assist humans in procedural tasks by deciding when to provide reminders"

**Slide 3: Task Domain**
- Use **Diagram 2 (Task Structure)** to show examples
- Explain: "Tasks vary from simple (8 steps) to complex (20 steps), with varying criticality"

**Slide 4-5: Problem Formulation**
- Use **Diagram 3 (Memory Dynamics)** to explain the failure model
- Explain: "Human memory decays, leading to failures. Reminders restore memory but cost attention. The challenge is optimal timing."

**Slide 6: Hypothesis**
- Use **Diagram 4 (Cost Strategy)** to set expectations
- Explain: "We hypothesize that different cost structures should lead to different learned strategies. Higher failure costs should produce more conservative (less interrupting) policies."

**Slide 7+: Results**
- Use existing results (training curves, performance comparisons, trajectory videos)
- Refer back to Diagram 4 to show RL learned the expected patterns

### For Papers

**Figure 1**: System Architecture (Diagram 1)
- Caption: "Procedure assistant system architecture showing the interaction between human agent, AI assistant, and environment. The assistant observes partial state and decides whether to provide reminders, receiving negative reward for both failures and interruptions."

**Figure 2**: Task Examples (Diagram 2)
- Caption: "Example procedural tasks with per-step criticality values. Gray steps are trivial (cannot fail), orange are critical (criticality=1.0), and red are ultra-critical (criticality>1.0). Failure costs scale with criticality."

**Figure 3**: Memory & Failure Model (Diagram 3)
- Caption: "Memory dynamics showing exponential decay (λ=0.05, 14-tick half-life) and restoration via interventions (δ=0.6 boost). Bottom panel shows resulting failure probability f(m) = f₀×exp(-k×m) with concrete examples: 60% failure at m=0 drops to 3% at m=1.0."

**Figure 4**: Strategy Regions (Diagram 4)
- Caption: "Cost regime configurations (left) and optimal strategy regions (right). Decision boundary shows when to intervene: p_fail > c_int/c_fail. Three regimes span different regions of the strategy space, leading to distinct learned behaviors."

---

## Technical Details

**Resolution**: 300 DPI (publication quality)
**Format**: PNG with transparent backgrounds where applicable
**Size**: 274 KB - 501 KB per diagram
**Tools**: matplotlib, seaborn, numpy
**Generation**: `python src/visualization/generate_explanation_diagrams.py`

---

## Customization

To modify diagrams:
1. Edit `src/visualization/generate_explanation_diagrams.py`
2. Run `python src/visualization/generate_explanation_diagrams.py`
3. Diagrams will be regenerated in `results/figures/`

**Common customizations**:
- Change tasks shown in Diagram 2 (edit `task1 = tasks['make_cereal']`)
- Adjust color schemes (edit color parameters)
- Add/remove annotations (edit text and arrow placement)
- Change resolution (edit `dpi=300` parameter)

---

## Complementary Visualizations

These diagrams explain the **problem formulation**. For **results**, use:
- `training_curves_all_models.png` - Shows learning progress
- `cross_task_performance_heatmap.png` - Shows which models succeeded
- `trajectory_*.gif` - Shows example episodes
- `cross_task_multi_regime_results.pptx` - Full results presentation

---

**Generated**: 2026-02-18
**Author**: Claude (Anthropic)
**Project**: Procedure Assistant RL Research
