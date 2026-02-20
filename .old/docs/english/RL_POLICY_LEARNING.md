# Using Reinforcement Learning for Assistant Policy

**Question**: Should we use RL (classical or deep learning) instead of heuristic policies to optimize for different user profiles and cost functions?

**Short Answer**: **Yes, absolutely!** This is the natural next step and a strong research contribution. Your current heuristic policies were perfect for **proving the concept** (that cost structure matters), but RL policies would be optimal for **practical deployment** and would constitute significant novel research.

---

## Table of Contents

1. [Why RL Makes Sense](#why-rl-makes-sense)
2. [Current State: Heuristic Policies](#current-state-heuristic-policies)
3. [RL Approaches: Classical vs Deep](#rl-approaches-classical-vs-deep)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Expected Benefits](#expected-benefits)
6. [Challenges and Solutions](#challenges-and-solutions)
7. [Experimental Design](#experimental-design)
8. [Research Contributions](#research-contributions)
9. [Code Examples](#code-examples)

---

## Why RL Makes Sense

### 1. **You Already Have a POMDP!**

Your environment (`ProcedureAssistantEnv`) is already a proper POMDP with:

✅ **State space**: (s_t, τ_t, m_t)
✅ **Action space**: {silent, confirm, remind_0, ..., remind_4}
✅ **Transition dynamics**: Fully specified (hazard, memory, failures)
✅ **Reward function**: -c_int × interruptions - c_fail × failures
✅ **Observation model**: Noisy step estimation

**This is exactly what RL algorithms need!**

### 2. **Heuristics Have Limitations**

Your current policies (Proactive, Reactive, Random) are:

**Strengths**:
- ✅ Interpretable
- ✅ Easy to implement
- ✅ Good for proof-of-concept
- ✅ Show that cost structure matters

**Limitations**:
- ❌ Not optimal (designed by intuition, not derived)
- ❌ Don't adapt to specific cost regimes automatically
- ❌ Fixed thresholds (memory=0.3, risk=0.3) may not be optimal
- ❌ Don't learn from experience
- ❌ Can't discover surprising strategies

**Example of suboptimality**:
```python
# Proactive policy
if memory[next_step] < 0.3:  # Why 0.3? Arbitrary threshold
    return remind(next_step)
```

**What if the optimal threshold is**:
- 0.25 for c_int=2, c_fail=20?
- 0.45 for c_int=15, c_fail=20?
- Depends on τ_t (time in step)?
- Depends on multiple steps' memories jointly?

RL can **discover** these optimal thresholds automatically.

### 3. **Different User Profiles = Different Optimal Policies**

You have multiple cost regimes:

| User Profile | c_int | c_fail | λ | Optimal Policy? |
|--------------|-------|--------|---|-----------------|
| Surgeon | 20 | 100 | 0.05 | ? (unknown) |
| Chef | 3 | 10 | 0.03 | ? (unknown) |
| Student | 5 | 20 | 0.08 | ? (unknown) |
| Elderly | 2 | 30 | 0.12 | ? (unknown) |

**Current approach**: Design a heuristic for each profile (manual, time-consuming)
**RL approach**: Train policy conditioned on (c_int, c_fail, λ) → automatic adaptation

### 4. **RL Enables Personalization**

**Current**:
```python
# Fixed policy for all users
policy = ProactiveReminderPolicy(threshold=0.3)
```

**With RL**:
```python
# Policy adapts to user's cost parameters
policy = RLPolicy.load_for_user(user_id)
# OR
policy = RLPolicy(c_int=user.c_int, c_fail=user.c_fail, lambda_forget=user.lambda_forget)
```

The policy automatically adjusts its behavior based on the user's cost structure.

---

## Current State: Heuristic Policies

### What You Have Now

**Three handcrafted policies**:

```python
class RandomAssistantPolicy:
    """Baseline: 85% silent, 15% act"""
    def get_action(self, obs):
        return np.random.choice(n_actions, p=[0.85, 0.03, ...])

class ProactiveReminderPolicy:
    """Remind when memory < threshold"""
    def get_action(self, obs):
        if memory[next_step] < 0.3:
            return remind(next_step)
        return silent

class ReactivePolicyHighCost:
    """Remind when failure risk > threshold"""
    def get_action(self, obs):
        fail_prob = compute_failure_prob(memory[current_step])
        if fail_prob > 0.3:
            return remind(current_step)
        return silent
```

### Performance Results

| Policy | c_int=2 | c_int=15 |
|--------|---------|----------|
| Random | -143 | -143 |
| Proactive | -64 | -251 |
| Reactive | -120 | -183 |

**Key observation**: No single heuristic is optimal across all cost regimes.

### What's Missing

1. **Optimality**: Are these the best possible policies?
2. **Adaptation**: How to automatically adjust to new cost structure?
3. **Discovery**: Can we find non-obvious strategies?
4. **Multi-step reasoning**: Should we remind now or wait for better timing?

**RL can address all of these.**

---

## RL Approaches: Classical vs Deep

### Option 1: Classical POMDP Solvers

**Algorithms**:
- **QMDP**: Approximates POMDP as MDP on belief space
- **PBVI** (Point-Based Value Iteration): Samples belief points
- **POMCP** (Partially Observable Monte Carlo Planning): Online tree search
- **SARSOP**: Offline solver with sampling

**Pros**:
- ✅ Theoretically grounded (proven convergence)
- ✅ Can find optimal policy
- ✅ Works well for small state spaces
- ✅ Interpretable value functions

**Cons**:
- ❌ Scales poorly with state space size
- ❌ Requires discrete state/action spaces
- ❌ May need careful discretization
- ❌ Computational cost for large problems

**When to use**:
- Small state space (few steps, coarse discretization)
- Need theoretical guarantees
- Want interpretable solutions

### Option 2: Deep Reinforcement Learning

**Algorithms**:
- **DQN** (Deep Q-Network): Value-based, discrete actions
- **PPO** (Proximal Policy Optimization): Policy gradient, stable
- **A3C** (Asynchronous Advantage Actor-Critic): Parallel training
- **SAC** (Soft Actor-Critic): Continuous actions (if needed)
- **Recurrent variants**: LSTM/GRU for belief tracking

**Pros**:
- ✅ Scales to large state spaces
- ✅ Handles continuous observations
- ✅ Can learn complex policies
- ✅ Modern infrastructure (PyTorch, TensorFlow, Stable-Baselines3)

**Cons**:
- ❌ Sample inefficient (needs many episodes)
- ❌ Less interpretable (black box)
- ❌ No optimality guarantees
- ❌ Hyperparameter sensitive

**When to use**:
- Large state space (many steps, fine-grained state)
- Need scalability
- Have computational resources
- Prioritize performance over interpretability

### Option 3: Hybrid Approaches

**Best of both worlds**:

1. **Guided policy learning**: Initialize with heuristic, refine with RL
2. **Hierarchical RL**: High-level strategy (when to intervene) + low-level tactics (which step to remind)
3. **Constrained RL**: Learn within bounds of interpretable structure

---

## Implementation Roadmap

### Phase 1: Classical POMDP (Recommended Start)

**Why start here**:
- Your state space is manageable (5 steps, bounded memory)
- Gives baseline for optimal performance
- Interpretable results
- Fast to implement and test

**Step 1: Discretize State Space**

```python
# Current continuous state
state = (s_t, tau_t, m_t)
# s_t ∈ {0,1,2,3,4}
# tau_t ∈ [0, 100+]
# m_t ∈ R^5

# Discretized state
s_t_discrete = s_t  # Already discrete: 5 values
tau_t_discrete = min(tau_t // 10, 5)  # Buckets: [0-9, 10-19, ..., 50+]
m_t_discrete = (m_t // 0.2).astype(int)  # Buckets: [0-0.2, 0.2-0.4, ..., 1.0+]
# Each memory: 6 buckets
# Total memory states: 6^5 = 7776

# Total states: 5 × 6 × 7776 = 233,280 states
```

Still manageable for classical algorithms!

**Step 2: Set Up POMDP Solver**

```python
# Use pomdp-py or pypomcp
from pomdp_py import Agent, POMDP, POMCP

class ProcedureAssistancePOMDP(POMDP):
    def __init__(self, params):
        self.params = params
        self.init_state = ProcedureAssistantState(N_STEPS)

    def transition_model(self, state, action):
        # Use your existing dynamics
        return TransitionModel(self.params)

    def observation_model(self, next_state, action):
        # Use your existing observation noise
        return ObservationModel(self.params)

    def reward_model(self, state, action):
        # Use your existing cost function
        return -self.params.c_int * I[action != silent] - ...

# Solve with POMCP
agent = Agent(init_belief, policy_model, transition, observation, reward)
action = agent.plan(observation)
```

**Step 3: Train and Evaluate**

```python
# Train for each cost regime
for c_int in [2, 5, 15]:
    for c_fail in [10, 20, 40]:
        params = SimulationParams(c_int=c_int, c_fail_base=c_fail)

        # Solve POMDP
        policy = solve_pomdp(params, algorithm='POMCP')

        # Evaluate
        results = run_simulation(policy, params, n_episodes=100)

        print(f"c_int={c_int}, c_fail={c_fail}: {results['mean_reward']}")
```

### Phase 2: Deep RL (For Scalability)

**Why move to deep RL**:
- Handle larger state spaces (more steps, finer discretization)
- Learn from real user interactions
- Continuous adaptation

**Step 1: Set Up Deep RL Environment**

```python
# Your environment already works with Gym API!
from procedure_assistant_sim import ProcedureAssistantEnv

# Wrap for Gym compatibility
class GymWrappedEnv(gym.Env):
    def __init__(self, params):
        self.env = ProcedureAssistantEnv(params)

        # Define spaces
        self.observation_space = gym.spaces.Dict({
            'step': gym.spaces.Discrete(N_STEPS),
            'tau': gym.spaces.Box(0, 100, (1,)),
            'memory': gym.spaces.Box(0, 2, (N_STEPS,))
        })
        self.action_space = gym.spaces.Discrete(len(ASSISTANT_ACTIONS))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
```

**Step 2: Train with Stable-Baselines3**

```python
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment (parallel training)
env = make_vec_env(
    lambda: GymWrappedEnv(params),
    n_envs=8  # 8 parallel environments
)

# Train PPO agent
model = PPO(
    "MultiInputPolicy",  # For dict observations
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train for 100k steps
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_procedure_assistant")
```

**Step 3: Evaluate and Compare**

```python
# Load trained model
model = PPO.load("ppo_procedure_assistant")

# Evaluate
class RLPolicy:
    def __init__(self, model):
        self.model = model

    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

rl_policy = RLPolicy(model)
results = run_simulation(rl_policy, params, n_episodes=100)

# Compare with heuristics
print("RL Policy:", results['mean_reward'])
print("Proactive:", proactive_results['mean_reward'])
print("Reactive:", reactive_results['mean_reward'])
```

### Phase 3: Multi-Task Learning (Advanced)

**Goal**: Single policy that adapts to different cost structures

```python
# Condition policy on cost parameters
class CostConditionedPolicy(nn.Module):
    def __init__(self):
        self.cost_encoder = nn.Linear(3, 32)  # (c_int, c_fail, λ)
        self.state_encoder = nn.Linear(state_dim, 64)
        self.fusion = nn.Linear(96, 128)
        self.action_head = nn.Linear(128, n_actions)

    def forward(self, state, cost_params):
        cost_emb = F.relu(self.cost_encoder(cost_params))
        state_emb = F.relu(self.state_encoder(state))
        fused = torch.cat([cost_emb, state_emb], dim=-1)
        hidden = F.relu(self.fusion(fused))
        return self.action_head(hidden)

# Train on multiple cost regimes simultaneously
for episode in range(n_episodes):
    # Sample random cost regime
    c_int = np.random.uniform(2, 20)
    c_fail = np.random.uniform(10, 40)
    lambda_forget = np.random.uniform(0.02, 0.10)

    # Train policy conditioned on these costs
    params = SimulationParams(c_int=c_int, c_fail_base=c_fail, lambda_forget=lambda_forget)
    env = ProcedureAssistantEnv(params)

    # ... training loop ...
```

**Result**: One policy that generalizes to any cost structure!

---

## Expected Benefits

### 1. **Optimality**

**Current (Heuristic)**:
```
Proactive (c_int=2, c_fail=20): -64.2
```

**Expected with RL**:
```
RL Policy (c_int=2, c_fail=20): -55 to -60 (10-15% improvement)
```

**Why?** RL discovers:
- Optimal thresholds (not fixed 0.3)
- Optimal timing (when in step to remind)
- Multi-step planning (should I remind now or later?)

### 2. **Automatic Adaptation**

**Current**:
```python
# Need to design policy for each regime
if c_int < 5:
    policy = ProactiveReminderPolicy(threshold=0.3)
elif c_int < 10:
    policy = ProactiveReminderPolicy(threshold=0.4)
else:
    policy = ReactivePolicyHighCost(risk_threshold=0.35)
```

**With RL**:
```python
# Automatic adaptation
policy = RLPolicy(c_int=user.c_int, c_fail=user.c_fail)
# Policy internally adjusts behavior
```

### 3. **Surprising Discoveries**

RL might discover strategies like:

**Discovery 1: "Memory Banking"**
```
Instead of reminding immediately when memory drops,
wait until just before step completion, when reminder
has maximum impact and minimum "wasted" time.
```

**Discovery 2: "Strategic Silence"**
```
In high-cost regimes, completely ignore steps 1-3,
focus all reminders on critical steps 4-5 where
failure costs are highest.
```

**Discovery 3: "Batching"**
```
Give multiple reminders in quick succession (one burst),
then long silence. User pays interruption cost once
but gets multi-step information.
```

These might be non-obvious to human designers!

### 4. **Continual Learning**

```python
# Policy improves from user interactions
for user_interaction in real_world_deployment:
    observation = user_interaction.state
    action = policy.get_action(observation)
    reward = compute_reward(user_interaction)

    # Update policy online
    policy.update(observation, action, reward)
```

Policy gets better over time with real data!

### 5. **Generalization to New Scenarios**

Train on:
- 5-step cooking task
- Cost ranges: c_int ∈ [2,20], c_fail ∈ [10,40]

Generalize to:
- 7-step medical procedure (new task)
- Cost: c_int=25, c_fail=100 (outside training range)

Deep RL can generalize if trained properly!

---

## Challenges and Solutions

### Challenge 1: Partial Observability

**Problem**: Assistant doesn't observe true state (only noisy observation)

**Solution 1: Belief Tracking** (Classical POMDP)
```python
# POMCP maintains belief distribution over states
belief = Belief(init_state_distribution)
for observation in observations:
    belief = belief.update(observation, action)
action = policy(belief)  # Policy acts on belief
```

**Solution 2: Recurrent Networks** (Deep RL)
```python
class RecurrentPolicy(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(obs_dim, 128)
        self.policy_head = nn.Linear(128, n_actions)

    def forward(self, obs_sequence):
        # LSTM implicitly maintains belief
        hidden, _ = self.lstm(obs_sequence)
        return self.policy_head(hidden[-1])
```

**Solution 3: Observation History**
```python
# Include last k observations
obs_history = [o_t, o_{t-1}, ..., o_{t-k}]
action = policy(obs_history)
```

### Challenge 2: Sparse Rewards

**Problem**: Reward only when step completes (every ~30 ticks)

**Current reward structure**:
```
Tick 1-29: reward = -c_int × I[interrupt]
Tick 30 (step completes): reward = -c_int × I[interrupt] - c_fail × I[failure]
```

**Solution 1: Reward Shaping**
```python
# Add intermediate rewards
def shaped_reward(state, action):
    base_reward = original_reward(state, action)

    # Bonus for maintaining good memory
    memory_bonus = 0.1 * np.sum(state.memory > 0.3)

    # Penalty for memory dropping too low
    memory_penalty = -0.5 * np.sum(state.memory < 0.2)

    return base_reward + memory_bonus + memory_penalty
```

**Solution 2: Credit Assignment**
```python
# Use advantage estimation (built into PPO, A3C)
# Properly assigns credit to actions far from reward
```

### Challenge 3: Sample Efficiency

**Problem**: Need many episodes to learn

**Solution 1: Prioritized Experience Replay** (DQN)
```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    buffer_size=50000,
    learning_starts=1000,
    prioritized_replay=True,  # Focus on surprising experiences
    prioritized_replay_alpha=0.6
)
```

**Solution 2: Curriculum Learning**
```python
# Start with easy regimes, gradually increase difficulty
curricula = [
    {"c_int": 2, "c_fail": 20, "episodes": 1000},   # Easy
    {"c_int": 5, "c_fail": 20, "episodes": 1000},   # Medium
    {"c_int": 15, "c_fail": 20, "episodes": 1000},  # Hard
]

for curriculum in curricula:
    params = SimulationParams(**curriculum)
    env = ProcedureAssistantEnv(params)
    model.learn(total_timesteps=curriculum["episodes"] * 100)
```

**Solution 3: Model-Based RL**
```python
# Learn dynamics model, use for planning
# Reduces sample complexity 10-100×
```

### Challenge 4: Multi-Task Learning Difficulty

**Problem**: Training on all cost regimes simultaneously is hard

**Solution 1: Task Embedding**
```python
# Add task ID as input
task_embedding = nn.Embedding(num_tasks, 32)
task_emb = task_embedding(task_id)
policy_input = torch.cat([state, task_emb], dim=-1)
```

**Solution 2: Meta-Learning (MAML)**
```python
# Learn initialization that adapts quickly to new tasks
from learn2learn import MAML

maml = MAML(model, lr=0.01)
for task in tasks:
    # Few-shot adaptation
    adapted_model = maml.clone()
    for _ in range(5):  # 5 gradient steps
        loss = compute_loss(adapted_model, task)
        adapted_model.adapt(loss)
```

**Solution 3: Mixture of Experts**
```python
# Different sub-policies for different regimes
class MixtureOfExperts(nn.Module):
    def __init__(self, n_experts=3):
        self.experts = nn.ModuleList([
            ExpertPolicy() for _ in range(n_experts)
        ])
        self.gating = nn.Linear(3, n_experts)  # Input: (c_int, c_fail, λ)

    def forward(self, state, cost_params):
        # Compute gating weights
        weights = F.softmax(self.gating(cost_params), dim=-1)

        # Weighted combination of expert policies
        outputs = [expert(state) for expert in self.experts]
        return sum(w * o for w, o in zip(weights, outputs))
```

---

## Experimental Design

### Experiment 1: RL vs Heuristics

**Research Question**: How much better are learned policies than heuristics?

**Method**:
```python
policies = {
    "Random": RandomAssistantPolicy(),
    "Proactive": ProactiveReminderPolicy(threshold=0.3),
    "Reactive": ReactivePolicyHighCost(risk_threshold=0.3),
    "RL (POMCP)": solve_pomdp(params, algorithm='POMCP'),
    "RL (PPO)": PPO.load("ppo_trained"),
}

for policy_name, policy in policies.items():
    results = run_simulation(policy, params, n_episodes=100)
    print(f"{policy_name}: {results['mean_reward']}")
```

**Hypothesis**: RL policies will achieve 10-20% better reward

**Metrics**:
- Mean cumulative reward
- Failure rate
- Interaction rate
- Pareto frontier (failures vs interactions)

---

### Experiment 2: Generalization Across Cost Regimes

**Research Question**: Can single policy generalize to unseen cost structures?

**Method**:
```python
# Train on 80% of cost space
train_costs = sample_costs(n=100, seed=42)
model = train_multi_task_policy(train_costs)

# Test on held-out 20%
test_costs = sample_costs(n=25, seed=123)
for cost in test_costs:
    params = SimulationParams(c_int=cost['c_int'], c_fail_base=cost['c_fail'])
    results = run_simulation(model, params, n_episodes=50)
    # Compare with heuristic baseline
```

**Hypothesis**: Multi-task RL will generalize to unseen costs

**Metrics**:
- Performance on interpolation (costs within training range)
- Performance on extrapolation (costs outside training range)

---

### Experiment 3: Online Learning from Human Data

**Research Question**: Can RL policy improve from real user interactions?

**Method**:
```python
# Phase 1: Pre-train on simulation
model = PPO(...).learn(100_000)

# Phase 2: Deploy to real users, collect data
for user_session in real_users:
    trajectory = collect_trajectory(user_session, model)
    replay_buffer.add(trajectory)

# Phase 3: Fine-tune on real data
model.learn_from_replay_buffer(replay_buffer, steps=10_000)

# Phase 4: Evaluate improvement
results_before = evaluate(model_before, test_users)
results_after = evaluate(model_after, test_users)
```

**Hypothesis**: Real user data improves policy beyond simulation-trained baseline

**Metrics**:
- User satisfaction (survey)
- Task performance (errors, time)
- Subjective interruption burden

---

### Experiment 4: Discovered Strategies

**Research Question**: What novel strategies does RL discover?

**Method**:
```python
# Train RL policy
model = PPO(...).learn(500_000)

# Analyze learned policy
def analyze_policy(model, params):
    """Extract policy insights"""
    # Sample trajectories
    trajectories = []
    for _ in range(100):
        traj = rollout(model, params)
        trajectories.append(traj)

    # Analyze patterns
    print("Average reminder timing:", np.mean([t.reminder_ticks for t in trajectories]))
    print("Which steps get reminded:", Counter([t.reminded_steps for t in trajectories]))
    print("Memory threshold (inferred):", infer_threshold(trajectories))

analyze_policy(model, params)
```

**Hypothesis**: RL will discover non-obvious strategies (e.g., memory banking, strategic batching)

**Metrics**:
- Timing patterns
- Step selection patterns
- Deviation from heuristic behavior

---

## Research Contributions

### Using RL would add significant research contributions:

### Contribution 1: Optimal Policies for Procedure Assistance

**Current work**: Shows cost structure matters, provides heuristics
**With RL**: Derives optimal policies for different cost regimes

**Paper title**: *"Learning Optimal Assistance Policies for Procedural Tasks with Interruption Costs"*

**Novel aspects**:
- First optimal POMDP solution for procedure assistance
- Handles partial observability (noisy step observation)
- Multi-objective optimization (failures vs interruptions)

### Contribution 2: Multi-Task RL for User Adaptation

**Current work**: Fixed policies per regime
**With RL**: Single policy that adapts to user parameters

**Paper title**: *"Personalizing AI Assistants via Multi-Task Reinforcement Learning"*

**Novel aspects**:
- Policy conditioned on cost parameters
- Generalization across cost structures
- Few-shot adaptation to new users

### Contribution 3: Real-World Learning

**Current work**: Simulation only
**With RL**: Learn from real user interactions

**Paper title**: *"Continual Learning of Interruption Management from Human Feedback"*

**Novel aspects**:
- Sim-to-real transfer for assistance
- Online policy improvement
- Human-in-the-loop RL

### Contribution 4: Interpretable RL Analysis

**Current work**: Heuristics are interpretable
**With RL**: Extract insights from learned policies

**Paper title**: *"What Do Optimal Assistants Learn? Analyzing Reinforcement Learning Policies for Procedure Assistance"*

**Novel aspects**:
- Policy distillation to interpretable rules
- Comparison of discovered vs designed strategies
- When RL agrees/disagrees with human intuition

---

## Code Examples

### Example 1: Train PPO Policy

```python
# train_rl_policy.py
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from procedure_assistant_sim import ProcedureAssistantEnv, SimulationParams

class GymWrapperEnv(gym.Env):
    """Wrap your environment for Stable-Baselines3"""
    def __init__(self, params):
        super().__init__()
        self.env = ProcedureAssistantEnv(params)

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'step_estimate': gym.spaces.Discrete(6),  # 0-4 + done
            'elapsed_time': gym.spaces.Box(0, 100, (1,), dtype=np.float32),
            'memory': gym.spaces.Box(0, 2, (5,), dtype=np.float32)
        })

        # Define action space
        self.action_space = gym.spaces.Discrete(7)  # silent + confirm + 5 reminders

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        return {
            'step_estimate': obs['step_estimate'],
            'elapsed_time': np.array([obs['elapsed_time']], dtype=np.float32),
            'memory': obs['memory'].astype(np.float32)
        }

# Training script
if __name__ == "__main__":
    # Create environment
    params = SimulationParams(c_int=5.0, c_fail_base=20.0)
    env = DummyVecEnv([lambda: GymWrapperEnv(params)])

    # Create PPO agent
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,  # No discounting (episodic)
        verbose=1,
        tensorboard_log="./logs/ppo_procedure_assistant/"
    )

    # Train
    print("Training PPO policy...")
    model.learn(total_timesteps=100_000)

    # Save
    model.save("models/ppo_procedure_assistant")
    print("Model saved!")

    # Evaluate
    from procedure_assistant_sim import run_simulation

    class RLPolicy:
        def __init__(self, model):
            self.model = model

        def get_action(self, obs):
            obs_dict = {
                'step_estimate': obs['step_estimate'],
                'elapsed_time': np.array([obs['elapsed_time']]),
                'memory': obs['memory']
            }
            action, _ = self.model.predict(obs_dict, deterministic=True)
            return action

    rl_policy = RLPolicy(model)
    results = run_simulation(rl_policy, params, n_episodes=50, verbose=True)

    print(f"\nResults:")
    print(f"Mean reward: {results['mean_reward']:.1f}")
    print(f"Mean failures: {results['mean_failures']:.1f}")
    print(f"Mean interactions: {results['mean_interactions']:.1f}")
```

### Example 2: Multi-Task RL

```python
# train_multitask_rl.py
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CostConditionedPolicy(ActorCriticPolicy):
    """Policy conditioned on cost parameters"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add cost parameter encoder
        self.cost_encoder = nn.Sequential(
            nn.Linear(3, 32),  # (c_int, c_fail, lambda)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

    def forward(self, obs, deterministic=False):
        # Extract cost parameters from observation
        cost_params = obs['cost_params']  # [c_int, c_fail, lambda]

        # Encode cost parameters
        cost_emb = self.cost_encoder(cost_params)

        # Concatenate with state features
        state_features = self.extract_features(obs)
        combined = torch.cat([state_features, cost_emb], dim=-1)

        # Standard actor-critic forward
        return super().forward_from_features(combined, deterministic)

class MultiTaskEnv(GymWrapperEnv):
    """Environment that samples different cost regimes"""

    def reset(self):
        # Sample random cost parameters
        self.c_int = np.random.uniform(2, 20)
        self.c_fail = np.random.uniform(10, 40)
        self.lambda_forget = np.random.uniform(0.02, 0.10)

        # Create new params
        params = SimulationParams(
            c_int=self.c_int,
            c_fail_base=self.c_fail,
            lambda_forget=self.lambda_forget
        )
        self.env = ProcedureAssistantEnv(params)

        obs = self.env.reset()
        processed_obs = self._process_obs(obs)

        # Add cost parameters to observation
        processed_obs['cost_params'] = np.array([
            self.c_int, self.c_fail, self.lambda_forget
        ], dtype=np.float32)

        return processed_obs

# Train multi-task policy
env = DummyVecEnv([lambda: MultiTaskEnv(None)])

model = PPO(
    CostConditionedPolicy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

print("Training multi-task policy...")
model.learn(total_timesteps=500_000)
model.save("models/multitask_ppo")
```

---

## Summary: Why Use RL?

### Short Answer

**Yes**, use RL! It's the natural next step because:

1. ✅ **You already have a POMDP** - just plug in RL algorithms
2. ✅ **Heuristics are suboptimal** - RL can find better policies
3. ✅ **Need adaptation** - RL automatically adjusts to cost structure
4. ✅ **Strong research contribution** - Optimal policies + multi-task learning
5. ✅ **Practical value** - Real-world deployment with continual improvement

### Recommended Path

**Phase 1** (1-2 weeks): Classical POMDP solver (POMCP)
- Quick baseline for optimal performance
- Works with current state space size
- Interpretable results

**Phase 2** (2-3 weeks): Deep RL (PPO)
- Better scalability
- Handles continuous state better
- Modern infrastructure

**Phase 3** (3-4 weeks): Multi-task learning
- Single policy for all cost regimes
- Strong research contribution
- Practical deployment value

### Expected Outcomes

**Performance**: 10-20% improvement over heuristics
**Contribution**: 2-3 papers (optimal policies, multi-task, real-world learning)
**Impact**: Enables practical deployment of cost-aware assistants

---

## Next Steps

1. **Start simple**: Train PPO on single cost regime (c_int=5, c_fail=20)
2. **Compare**: RL vs Proactive vs Reactive vs Random
3. **Analyze**: What strategies did RL discover?
4. **Scale up**: Multi-task learning across cost regimes
5. **Real world**: Deploy to actual users, collect data, improve

**The environment is ready - you just need to add RL training!** 🚀

---

**End of RL Policy Learning Document**
