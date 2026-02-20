# Intervention Success Videos - Index

This directory contains 10 animated GIF visualizations showing RL agents learning context-dependent intervention strategies across different tasks and cost regimes.

## How to Interpret the Videos

Each GIF shows:
- **Left Panel**: Task progression in circular layout
  - Blue line: Agent's path through the task
  - Blue circle: Current position
  - **Red stars (★)**: Intervention points (when agent provides reminders)
  - **Red X**: Failure events
  - Step colors indicate criticality (darker = more critical)

- **Right Panel**: Memory evolution heatmap
  - Rows: Each step in the procedure
  - Columns: Time progression
  - Color intensity: Memory level (darker = better memory)
  - Shows how memory decays and is restored by interventions

## Videos Organized by Strategy

### 🎯 Strategic Silence (0-1 interventions/episode)

#### 1. **make_sandwich_moderate_low.gif** (BEST PERFORMER) ⭐
- **Performance**: +29.96% improvement, 0 interventions
- **Why it works**: Low stakes (c_fail=10) → interruption cost > failure prevention benefit
- **Key insight**: Agent learned perfect strategic silence - knows when NOT to intervene
- **Video shows**: Clean trajectory with no interventions, successful completion

#### 2. **make_sandwich_balanced.gif**
- **Performance**: +27.48% improvement, 0 interventions
- **Why it works**: Simple 9-step task with low baseline failure rate
- **Key insight**: Even in balanced regime, strategic silence wins for simple tasks
- **Video shows**: Similar to above - no interventions needed

---

### 🎯 Sparse Strategic Interventions (2-7 interventions/episode)

#### 3. **latte_making_very_high_stakes.gif** (HIGH STAKES, SPARSE) ⭐
- **Performance**: +28.35% improvement, 2.18 interventions
- **Why it works**: Very high stakes (c_fail=30) but agent still conservative
- **Key insight**: Learned to intervene ONLY at truly critical moments despite high failure costs
- **Video shows**: 20-step complex task, ~2 well-timed interventions at critical steps
- **Watch for**: Red stars appearing at "brew_coffee" or "steam_milk" (criticality=1.0) when memory is low

#### 4. **make_stencil_balanced.gif** (CRAFTING TASK)
- **Performance**: +23.61% improvement, 4.43 interventions
- **Why it works**: Safety-critical steps (laser cutting) require targeted interventions
- **Key insight**: Learned which steps are dangerous vs. safe
- **Video shows**: 17-step procedure with 4-5 strategic interventions at high-risk steps
- **Watch for**: Interventions during laser operation or painting steps

#### 5. **make_coffee_balanced.gif** (PERFECT BALANCE) ⭐
- **Performance**: +28.42% improvement, 6.22 interventions
- **Why it works**: Balanced cost structure enables optimal intervention-silence tradeoff
- **Key insight**: Agent learned precise timing - intervenes when memory < critical threshold
- **Video shows**: 8-step task with 6-7 well-placed interventions
- **Watch for**: Memory heatmap showing interventions restore memory just before critical steps

---

### 🎯 Moderate Interventions (15-30 interventions/episode)

#### 6. **make_stencil_very_high_stakes.gif**
- **Performance**: +28.38% improvement, 16.39 interventions
- **Why it works**: High stakes + complex task → more interventions justified
- **Key insight**: More interventions than sparse strategy, but still selective
- **Video shows**: 17-step task with moderate intervention frequency
- **Watch for**: Interventions clustered around safety-critical steps

#### 7. **make_cereal_very_high_stakes.gif**
- **Performance**: +19.20% improvement, 22.19 interventions
- **Why it works**: High failure costs drive more cautious behavior
- **Key insight**: Shows regime effect - same task would have fewer interventions in low-stakes regime
- **Video shows**: 8-step simple task with frequent interventions

#### 8. **make_tea_very_high_stakes.gif**
- **Performance**: +9.30% improvement, 23.70 interventions
- **Why it works**: Moderate success with frequent intervention approach
- **Video shows**: 9-step task with interventions throughout

#### 9. **latte_making_balanced.gif** (COMPLEX TASK)
- **Performance**: +15.94% improvement, 26.34 interventions
- **Why it works**: 20-step procedure requires more oversight
- **Key insight**: Task complexity drives intervention frequency
- **Video shows**: Longest task (20 steps) with distributed interventions

---

### ⚠️ Over-Intervention (Failure Case)

#### 10. **cooking_balanced.gif** (LEARNED DYSFUNCTION)
- **Performance**: -21.36% improvement (WORSE than random!), 68.58 interventions
- **Why it failed**: Agent learned to over-intervene, not strategic behavior
- **Key insight**: Example of when RL fails - learned maladaptive policy
- **Video shows**: 14-step task with excessive interventions (nearly 5 per step!)
- **Watch for**: Dense red stars throughout - agent never learned to be silent

---

## Key Patterns Across Videos

### 1. **Strategic Silence Success** (Videos 1-2)
- **When**: Low-stakes regimes + simple tasks
- **Behavior**: 0 interventions, relies on human's baseline competence
- **Memory pattern**: Decays naturally but stays above failure threshold
- **Outcome**: Best improvement scores (+28-30%)

### 2. **Targeted Critical Interventions** (Videos 3-5)
- **When**: Balanced/high-stakes regimes with identifiable critical steps
- **Behavior**: 2-7 interventions at specific high-risk moments
- **Memory pattern**: Interventions restore memory just before critical steps
- **Outcome**: Strong improvement (+23-28%), low overhead

### 3. **Regime-Driven Frequency** (Videos 6-9)
- **When**: High-stakes regimes or complex tasks
- **Behavior**: 15-30 interventions spread across procedure
- **Memory pattern**: Frequent memory restoration throughout
- **Outcome**: Moderate improvement (+9-28%), higher intervention cost

### 4. **Learning Failure** (Video 10)
- **When**: Complex task with insufficient exploration or bad hyperparameters
- **Behavior**: Excessive interventions (>50), no strategic pattern
- **Memory pattern**: Chaotic, no clear intervention logic
- **Outcome**: Negative improvement, learned dysfunction

---

## What Makes an Intervention "Successful"?

An intervention successfully prevents failure when:

1. **Timing**: Occurs when memory is low (<0.4) at a critical step
2. **Criticality**: Step has high failure cost (criticality > 0)
3. **Outcome**: Step completes without failure after intervention
4. **Efficiency**: Minimal intervention overhead (cost-benefit positive)

**Best examples**: Videos 3-5 show this perfectly - sparse interventions precisely when needed.

---

## How the Agent Learned These Strategies

**Training**: 200,000 timesteps with PPO algorithm
- Reward signal: -(failures × c_fail + interventions × c_int)
- Observation: Memory state for all steps + noisy step estimate
- Action: Silent (0) or remind specific step (1-N)

**Emergent behaviors**:
- **High stakes** (c_fail >> c_int) → "Better safe than sorry" → More interventions
- **Low stakes** (c_fail ≈ c_int) → "Don't interrupt" → Strategic silence
- **Complex tasks** → More interventions needed for oversight
- **Critical steps** → Learned to focus on high-criticality moments

---

## Technical Details

**Video format**: Animated GIF, 10 FPS
**Generation**: `visualize_episode_trajectory.py --task {task} --regime {regime} --animate`
**Episode length**: Varies by task (8-20 steps × 30-50 ticks/step = 240-1000 ticks)
**File sizes**: 80KB - 867KB depending on task complexity

**Visualization code**: `/home/ec2-user/prism-sim/src/visualization/visualize_episode_trajectory.py`

---

## Recommended Viewing Order

**For understanding RL success**:
1. make_sandwich_moderate_low.gif - Perfect strategic silence
2. make_coffee_balanced.gif - Perfect balance
3. latte_making_very_high_stakes.gif - Sparse in high-stakes
4. cooking_balanced.gif - What failure looks like (contrast)

**For understanding intervention strategies**:
1. make_sandwich_moderate_low.gif (0 interventions)
2. latte_making_very_high_stakes.gif (2 interventions)
3. make_coffee_balanced.gif (6 interventions)
4. make_stencil_very_high_stakes.gif (16 interventions)
5. cooking_balanced.gif (68 interventions - failure)

**For understanding regime effects**:
- Compare make_cereal across regimes (if available)
- Compare make_stencil: balanced (4) vs very_high_stakes (16) interventions

---

**Last updated**: 2026-02-18
**Total videos**: 10
**Total file size**: ~3.3 MB
