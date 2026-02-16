# Procedure Assistant Simulation

**A lightweight POMDP simulator for studying human-AI collaboration in procedural tasks**

Based on the formulation in `modeling.pdf` — implements memory dynamics, interaction costs, and partial observability.

---

## Quick Start

### Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Run all experiments (~60 seconds)
python run_experiments.py
```

### View Results

**Plots**: `results_exp*.png` and `trajectory_*.png`
**Data**: `experiment_results.json`
**Analysis**: `IMPLEMENTATION_REFLECTION.md` (detailed design decisions)
**Summary**: `SUMMARY.md` (key findings)

---

## Core Concepts

### The POMDP

**State**: `x_t = (s_t, tau_t, m_t)`
- `s_t`: Current procedural step (0-4 or done)
- `tau_t`: Elapsed time in current step
- `m_t`: Memory vector (one value per step)

**Actions** (Assistant):
- `silent`: Do nothing
- `confirm`: Ask user for current step
- `remind_n`: Provide reminder for step n

**Actions** (Human):
- `silent`: Continue task
- `narrate`: Proactively share info
- `respond`: Reply to assistant query

**Observations**: Noisy estimate of current step + elapsed time

**Rewards**: `-c_int * I[interrupt] - c_fail * I[failure]`

### Key Parameters

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| Interruption cost | `c_int` | 5.0 | Cost when assistant speaks |
| Failure cost | `c_fail` | 20.0 | Cost when step fails |
| Forgetting rate | `lambda_forget` | 0.05 | Memory decay per tick |
| Reminder boost | `delta_reminder` | 0.3 | Memory increase from reminder |
| Observation noise | `obs_noise` | 0.2 | Prob. of wrong step estimate |
| Base failure prob | `f0_base` | 0.3 | Failure prob. with no memory |
| Memory effect | `k_memory` | 2.0 | How much memory reduces failure |

---

## Usage Examples

### Example 1: Run Single Simulation

```python
from procedure_assistant_sim import *

# Define parameters
params = SimulationParams(
    c_int=10.0,        # High interruption cost
    c_fail=30.0,       # High failure cost
    lambda_forget=0.05
)

# Choose policy
policy = ProactiveReminderPolicy(memory_threshold=0.3)

# Run simulation
results = run_simulation(policy, params, n_episodes=50, verbose=True)

# Print results
print(f"Mean reward: {results['mean_reward']:.1f}")
print(f"Mean failures: {results['mean_failures']:.1f}")
print(f"Mean interactions: {results['mean_interactions']:.1f}")
```

### Example 2: Custom Policy

```python
class MyCustomPolicy:
    """
    Example: Remind about next step if memory is low AND
    we're close to completing current step
    """
    def __init__(self, memory_threshold=0.3, time_threshold=20):
        self.memory_threshold = memory_threshold
        self.time_threshold = time_threshold

    def get_action(self, obs):
        current_step = obs['step_estimate']
        elapsed_time = obs['elapsed_time']
        memory = obs['memory']

        # If done, stay silent
        if obs['step_name'] == 'done':
            return ASSISTANT_ACTIONS['silent']

        # If current step has been going for a while
        if elapsed_time > self.time_threshold:
            # Check if next step has low memory
            next_step = current_step + 1
            if next_step < len(PROCEDURAL_STEPS):
                if memory[next_step] < self.memory_threshold:
                    return ASSISTANT_ACTIONS[f'remind_{next_step}']

        return ASSISTANT_ACTIONS['silent']

# Use it
policy = MyCustomPolicy(memory_threshold=0.25, time_threshold=15)
params = SimulationParams(c_int=8.0, c_fail=25.0)
results = run_simulation(policy, params, n_episodes=30)
```

### Example 3: Visualize Single Episode

```python
from procedure_assistant_sim import *

params = SimulationParams(c_int=5.0, c_fail=20.0)
policy = ProactiveReminderPolicy(memory_threshold=0.3)

# Run one episode
env = ProcedureAssistantEnv(params)
obs = env.reset()
done = False

while not done:
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action)

# Visualize trajectory
plot_trajectory(env.history, 'my_trajectory.png')
print("Trajectory saved to my_trajectory.png")

# Print episode statistics
print(f"Total reward: {sum(env.history['rewards']):.1f}")
print(f"Total failures: {env.pa_state.total_failures}")
print(f"Total interactions: {env.pa_state.total_interactions}")
print(f"Episode length: {len(env.history['rewards'])} ticks")
```

### Example 4: Parameter Sweep

```python
import numpy as np
import matplotlib.pyplot as plt

# Sweep over interruption cost
c_int_values = [2, 5, 8, 12, 15, 20]
policy = ProactiveReminderPolicy(memory_threshold=0.3)

rewards = []
failures = []
interactions = []

for c_int in c_int_values:
    params = SimulationParams(c_int=c_int, c_fail=20.0)
    results = run_simulation(policy, params, n_episodes=20, verbose=False)
    rewards.append(results['mean_reward'])
    failures.append(results['mean_failures'])
    interactions.append(results['mean_interactions'])
    print(f"c_int={c_int}: reward={results['mean_reward']:.1f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(c_int_values, rewards, marker='o')
axes[0].set_xlabel('Interruption Cost')
axes[0].set_ylabel('Mean Reward')
axes[0].set_title('Reward vs Interruption Cost')
axes[0].grid(alpha=0.3)

axes[1].plot(c_int_values, failures, marker='o', color='red')
axes[1].set_xlabel('Interruption Cost')
axes[1].set_ylabel('Mean Failures')
axes[1].set_title('Failures vs Interruption Cost')
axes[1].grid(alpha=0.3)

axes[2].plot(c_int_values, interactions, marker='o', color='blue')
axes[2].set_xlabel('Interruption Cost')
axes[2].set_ylabel('Mean Interactions')
axes[2].set_title('Interactions vs Interruption Cost')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_sweep.png', dpi=150)
print("Sweep plot saved to parameter_sweep.png")
```

---

## Understanding the Code

### File Structure

```
procedure_assistant_sim.py (600 lines)
├── PROCEDURAL_STEPS            # Task definition (5 steps)
├── ProcedureAssistantState     # State tracking (s_t, tau_t, m_t)
├── SimulationParams            # Cost structure and parameters
├── ProcedureAssistantEnv       # Main POMDP environment
│   ├── reset()                 # Start new episode
│   ├── step(action)            # Execute one tick
│   ├── _get_observation()      # Generate noisy obs
│   ├── _update_memory()        # Apply Eq. 3 from paper
│   ├── _compute_failure_probability()  # Apply Eq. 4
│   ├── _check_step_completion()        # Discrete hazard model
│   └── _sample_human_action()          # Bounded rationality (Eqs. 5-6)
├── Policy Classes
│   ├── RandomAssistantPolicy   # Baseline
│   ├── ProactiveReminderPolicy # Memory-based reminders
│   └── ReactivePolicyHighCost  # Risk-based reminders
├── run_simulation()            # Multi-episode runner
└── Visualization Functions
    ├── plot_results()          # Aggregate comparison
    └── plot_trajectory()       # Single episode

run_experiments.py (370 lines)
├── experiment_1_cost_comparison()
├── experiment_2_policy_comparison()
├── experiment_3_failure_cost_tradeoff()
├── experiment_4_memory_dynamics()
├── visualize_sample_trajectories()
└── save_summary_report()
```

### Key Methods

**`env.step(action)`**: Execute one tick
- Input: Assistant action (0-6)
- Output: `(observation, reward, done, info)`
- Side effects: Updates state, memory, samples failures

**`policy.get_action(obs)`**: Choose action
- Input: Observation dict with keys: `step_estimate`, `elapsed_time`, `memory`, `step_name`
- Output: Action ID (0-6)

**`run_simulation(policy, params, n_episodes)`**: Run multiple episodes
- Returns dict with keys: `mean_reward`, `mean_failures`, `mean_interactions`, `histories`

---

## Key Findings

### 1. Interruption Cost Dominates

When `c_int` is high (≥10), interaction costs overwhelm failure prevention benefits.

**Implication**: Real assistants in high-attention contexts (surgery, driving) must be extremely conservative.

### 2. Random Beats Smart (Sometimes)

Under high interruption cost, a mostly-silent random policy outperforms sophisticated reasoning.

**Why?**: Avoiding interruptions matters more than preventing failures.

### 3. Context Is Everything

Optimal policy depends on cost ratio (`c_fail / c_int`):
- High ratio: Be proactive
- Low ratio: Be minimalist
- Medium ratio: Be reactive

No universal "best" assistant.

### 4. Memory Dynamics Matter

Forgetting rate (`lambda`) determines optimal reminder frequency:
- Slow forgetting → fewer reminders
- Fast forgetting → more reminders

Should personalize to individual memory characteristics.

---

## Extending the Simulation

### Add New Policies

```python
class BeliefTrackerPolicy:
    """
    Maintains belief distribution over current state
    and chooses action based on expected value
    """
    def __init__(self, params):
        self.params = params
        self.belief = None  # Belief state (probability distribution)

    def reset(self):
        # Initialize uniform belief
        self.belief = np.ones(N_STEPS) / N_STEPS

    def update_belief(self, obs, action):
        # Bayesian update based on observation
        # (implement observation model and state transition)
        pass

    def get_action(self, obs):
        # Compute expected value of each action
        # Return action with highest value
        pass
```

### Add New Observations

Modify `_get_observation()` in `ProcedureAssistantEnv`:

```python
def _get_observation(self):
    # Existing: noisy step + elapsed time
    obs = {...}

    # Add new modality: confidence level
    obs['confidence'] = 1.0 - self.params.obs_noise

    # Add task-specific features
    obs['completion_fraction'] = self.pa_state.current_step / N_STEPS

    return obs
```

### Add New Cost Functions

Subclass `SimulationParams`:

```python
class DynamicCostParams(SimulationParams):
    """
    Interruption cost increases with each interaction
    (models user frustration)
    """
    def __init__(self, c_int_base=5.0, c_int_growth=0.5, **kwargs):
        super().__init__(c_int=c_int_base, **kwargs)
        self.c_int_base = c_int_base
        self.c_int_growth = c_int_growth
        self.interaction_count = 0

    def get_interruption_cost(self):
        cost = self.c_int_base + self.c_int_growth * self.interaction_count
        self.interaction_count += 1
        return cost

    def reset(self):
        self.interaction_count = 0
```

Then modify reward calculation in `step()` to use `params.get_interruption_cost()`.

### Add Multi-Step Lookahead

```python
class PlanningPolicy:
    """
    Plans ahead multiple steps using value iteration
    """
    def __init__(self, horizon=3):
        self.horizon = horizon

    def simulate_trajectory(self, state, action_sequence):
        # Roll out state dynamics for action sequence
        # Return expected cumulative reward
        pass

    def get_action(self, obs):
        # Try all action sequences of length self.horizon
        # Return first action of best sequence
        pass
```

---

## Common Issues

### Issue 1: Policy returns invalid action

**Symptom**: `KeyError` or `IndexError`
**Cause**: Policy returns step index outside [0, N_STEPS)
**Fix**: Check bounds before returning `ASSISTANT_ACTIONS[f'remind_{step}']`

```python
# Bad
return ASSISTANT_ACTIONS[f'remind_{next_step}']

# Good
if next_step < N_STEPS:
    return ASSISTANT_ACTIONS[f'remind_{next_step}']
else:
    return ASSISTANT_ACTIONS['silent']
```

### Issue 2: Episode never terminates

**Symptom**: Simulation hangs
**Cause**: Step completion probability is 0 or very low
**Fix**: Check hazard function in `_check_step_completion()`, ensure hazard increases with time

### Issue 3: Memory values explode

**Symptom**: Memory > 10 after many reminders
**Cause**: Missing clipping in `_update_memory()`
**Fix**: Already implemented—memory clipped to [0, 2.0]

### Issue 4: All failures or no failures

**Symptom**: Unrealistic failure rates (0% or 100%)
**Cause**: Wrong parameters in failure model
**Fix**: Check `f0_base` ∈ [0.2, 0.5] and `k_memory` ∈ [1.0, 3.0]

---

## Performance Tips

### Speed Up Simulations

1. **Reduce episode count**: Use `n_episodes=10` for quick tests
2. **Shorten episodes**: Decrease `step_mean_duration` to 20
3. **Disable plotting**: Set `verbose=False` in `run_simulation()`
4. **Parallelize**: Run multiple conditions in parallel using `multiprocessing`

```python
from multiprocessing import Pool

def run_condition(c_int):
    params = SimulationParams(c_int=c_int)
    policy = ProactiveReminderPolicy()
    return run_simulation(policy, params, n_episodes=20, verbose=False)

with Pool(4) as p:
    results = p.map(run_condition, [2, 5, 10, 15])
```

---

## Citation

If you use this code, please cite the formulation paper:

```
@article{arakawa2026procedure,
  title={Modeling Procedure Assistance},
  author={Arakawa, Riku},
  journal={Unpublished manuscript},
  year={2026}
}
```

And acknowledge this implementation:

```
Implementation by Claude (Anthropic), February 2026
Repository: [your repo URL]
```

---

## Contact

For questions or extensions:
- See `IMPLEMENTATION_REFLECTION.md` for design rationale
- See `SUMMARY.md` for experimental findings
- Modify `procedure_assistant_sim.py` directly for custom behavior

---

## License

[Your license choice]

---

**Happy simulating! 🎯**
