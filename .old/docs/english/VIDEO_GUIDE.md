# Kitchen Visualization Videos - Guide

**Generated**: February 14, 2026
**Purpose**: Visualize human-AI collaboration dynamics in a kitchen setting across different cost configurations

---

## 🎬 What You're Watching

Each video shows:

### **Left Panel: Kitchen Layout (Top-Down View)**
- **Blue circle**: Human agent performing the cooking task
- **Stations**:
  - 🟡 ONION dispenser (top-left)
  - 🟤 POT (center)
  - ⬜ DISH dispenser (top-right)
  - 🟨 SERVE area (bottom-right)
- **Red notifications**: Assistant reminders/confirmations (speech bubbles)
- **Red X marks**: Failures when they occur
- **Bottom label**: Current procedural step

### **Top-Right Panel: Memory State**
- **Horizontal bars**: Memory level for each step (0.0 to 1.0)
- **Green bar**: Current step being executed
- **Gray bars**: Other steps
- **Red dashed line**: Memory threshold (typically 0.3)
- Memory decays over time and boosts when assistant reminds

### **Bottom-Right Panel: Cost Tracking**
- **Total Reward**: Cumulative reward (blue if positive, red if negative)
- **Failures**: Total failure cost accumulated (always negative)
- **Interactions**: Total interruption cost accumulated (always negative)
- Values update in real-time as episode progresses

### **Title Bar**
- Displays: Policy name + cost parameters (c_int, c_fail, λ)

---

## 📹 Video Descriptions

### **Video 1: Low Cost Proactive** (1.5 seconds)
**File**: `video1_low_cost_proactive.mp4`

**Configuration**:
- c_int = 2.0 (LOW interruption cost)
- c_fail = 20.0
- λ = 0.05 (medium forgetting)
- Policy: Proactive (reminds when memory < 0.3)

**Episode Results**:
- Total Reward: -56.0
- Failures: 1
- Interactions: 10

**What to Observe**:
- ✓ **Frequent reminders**: Assistant sends ~10 notifications throughout
- ✓ **Low interaction cost**: Each interruption only costs 2, so total = 20
- ✓ **Good memory maintenance**: Memory bars stay mostly above threshold
- ✓ **Only 1 failure**: Reminders effectively prevent mistakes

**Key Insight**: When interruptions are cheap, proactive reminding works well. The assistant can afford to be "chatty" without overwhelming the cost structure.

---

### **Video 2: High Cost Proactive** (2.7 seconds)
**File**: `video2_high_cost_proactive.mp4`

**Configuration**:
- c_int = 15.0 (HIGH interruption cost)
- c_fail = 20.0
- λ = 0.05 (medium forgetting)
- Policy: Proactive (reminds when memory < 0.3)

**Episode Results**:
- Total Reward: -221.0
- Failures: 1
- Interactions: 11

**What to Observe**:
- ⚠️ **Same reminder frequency**: Still sends ~11 notifications (policy unchanged)
- ⚠️ **Massive interaction cost**: 11 × 15 = 165 cost from interruptions alone
- ⚠️ **Still only 1 failure**: Prevents mistakes, but at huge cost
- ⚠️ **3.9× worse reward**: Total reward degrades dramatically (-56 → -221)

**Key Insight**: **The Interruption Cost Paradox in action!** Same behavior, same failure prevention, but performance collapses when interruptions become expensive. The policy doesn't adapt to the cost context.

---

### **Video 3: High Cost Reactive** (1.5 seconds)
**File**: `video3_high_cost_reactive.mp4`

**Configuration**:
- c_int = 15.0 (HIGH interruption cost)
- c_fail = 20.0
- λ = 0.05 (medium forgetting)
- Policy: Reactive (reminds only when failure risk > 0.30)

**Episode Results**:
- Total Reward: -118.0
- Failures: 1
- Interactions: 5

**What to Observe**:
- ✓ **Fewer reminders**: Only 5 notifications (vs. 11 in Video 2)
- ✓ **Much lower interaction cost**: 5 × 15 = 75 (vs. 165)
- ✓ **Same failures**: Still 1 failure (similar to Video 2)
- ✓ **Better overall reward**: -118 vs. -221 (46% improvement!)

**Key Insight**: **Adaptation matters!** By waiting until failure risk is elevated, the reactive policy achieves similar failure prevention with far fewer costly interruptions. This is the right strategy for high-cost contexts.

**Comparison with Video 2**:
| Metric | Video 2 (Proactive) | Video 3 (Reactive) | Winner |
|--------|---------------------|-------------------|--------|
| Interactions | 11 | 5 | Reactive (54% less) |
| Failures | 1 | 1 | Tie |
| Total Reward | -221 | -118 | **Reactive (46% better)** |

---

### **Video 4: High Failure Cost** (2.4 seconds)
**File**: `video4_high_fail_cost.mp4`

**Configuration**:
- c_int = 3.0 (LOW interruption cost)
- c_fail = 40.0 (HIGH failure cost)
- λ = 0.05 (medium forgetting)
- Policy: Aggressive Proactive (memory threshold 0.25, lookahead 2)

**Episode Results**:
- Total Reward: -153.0
- Failures: 2
- Interactions: 13

**What to Observe**:
- ⚠️ **Very frequent reminders**: 13 interruptions (most of any video)
- ⚠️ **Low interruption cost**: 13 × 3 = 39 (acceptable given cheap interruptions)
- ⚠️ **Still 2 failures**: Even aggressive reminding can't eliminate all failures
- ⚠️ **High failure penalty**: 2 × 40 = 80 cost from mistakes

**Key Insight**: When failures are expensive (e.g., surgery, chemistry), you MUST be proactive even if it means frequent interruptions. The cost of a mistake outweighs the annoyance of reminders.

**Design Implication**: High-stakes domains (medical, safety-critical) should have aggressive assistants. Don't optimize for "user comfort" when lives are at stake.

---

### **Video 5: Fast Forgetting** (3.1 seconds)
**File**: `video5_fast_forgetting.mp4`

**Configuration**:
- c_int = 5.0 (medium interruption cost)
- c_fail = 20.0
- λ = 0.10 (HIGH forgetting rate)
- Policy: Proactive (memory threshold 0.35, adjusted)

**Episode Results**:
- Total Reward: -163.0
- Failures: 0
- Interactions: 26 (!)

**What to Observe**:
- 🔴 **Extreme reminder frequency**: 26 interruptions (highest of all videos)
- ✓ **No failures**: Perfect task execution (memory never drops too low)
- ⚠️ **Memory decays rapidly**: Watch memory bars in right panel - they drop quickly
- ⚠️ **High interaction cost**: 26 × 5 = 130 from interruptions alone

**Key Insight**: **Memory dynamics dominate strategy.** When users forget quickly (λ=0.10, half-life ~7 ticks), the assistant must remind very frequently to maintain memory above safety threshold. This creates an "interruption treadmill" where you're constantly fighting forgetting.

**Design Implication**:
- For tasks with long delays (days/weeks between steps): Need spaced repetition
- For users with poor working memory: Need more frequent reminders
- Should personalize λ based on individual memory characteristics

---

## 🔄 Key Comparisons

### **Same Policy, Different Costs** (Videos 1 vs. 2)

| Video | c_int | Interactions | Failures | Reward | Interpretation |
|-------|-------|--------------|----------|--------|----------------|
| 1 | 2 | 10 | 1 | -56 | Proactive works well when cheap |
| 2 | 15 | 11 | 1 | -221 | Same policy fails when expensive |

**Lesson**: A "good" policy in one cost regime can be terrible in another. Context-awareness is critical.

---

### **Same Costs, Different Policies** (Videos 2 vs. 3)

| Video | Policy | Interactions | Failures | Reward | Strategy |
|-------|--------|--------------|----------|--------|----------|
| 2 | Proactive | 11 | 1 | -221 | Remind frequently |
| 3 | Reactive | 5 | 1 | -118 | Remind only when risky |

**Lesson**: Under high interruption cost, reactive > proactive. Wait until failure risk is elevated before intervening.

---

### **Different Priorities** (Videos 1 vs. 4)

| Video | Scenario | c_fail/c_int | Interactions | Strategy |
|-------|----------|--------------|--------------|----------|
| 1 | Casual cooking | 10.0 | 10 | Moderate reminders |
| 4 | High-stakes | 13.3 | 13 | Aggressive prevention |

**Lesson**: High-stakes domains justify aggressive reminders. Don't optimize for "comfort" when mistakes are catastrophic.

---

### **Individual Differences** (Video 5)

| Metric | Normal (λ=0.05) | Fast Forget (λ=0.10) | Ratio |
|--------|-----------------|---------------------|-------|
| Interactions | 10-13 | 26 | 2× more |
| Failures | 1 | 0 | Better |
| Memory decay rate | 5%/tick | 10%/tick | 2× faster |

**Lesson**: Memory characteristics matter. Fast forgetters need 2× more reminders to achieve same performance. Should personalize based on user.

---

## 🎯 How to Interpret the Visualizations

### **Watch for These Patterns**:

1. **Memory Decay**:
   - Watch the right panel - memory bars shrink over time (exponential decay)
   - Faster in Video 5 (λ=0.10) than Videos 1-4 (λ=0.05)

2. **Reminder Timing**:
   - Proactive: Reminds before memory hits threshold (preventive)
   - Reactive: Reminds when failure risk is already elevated (reactive)
   - Notice timing difference between Videos 2 and 3

3. **Cost Accumulation**:
   - Bottom-right panel shows costs building up
   - Video 2: Interaction cost dominates
   - Video 4: Failure cost dominates
   - Notice which bar grows faster

4. **Agent Movement**:
   - Blue circle moves through kitchen: Onion → Pot → Dish → Serve
   - Movement reflects procedural step progression
   - Failures shown as red X at current location

5. **Notification Frequency**:
   - Count red speech bubbles that appear
   - Video 1-2: ~10-11 reminders
   - Video 3: ~5 reminders (more selective)
   - Video 5: ~26 reminders (fighting forgetting)

---

## 📊 Quantitative Summary

| Video | Cost Regime | Policy | Duration | Interactions | Failures | Reward |
|-------|-------------|--------|----------|--------------|----------|--------|
| 1 | Low int | Proactive | 1.5s | 10 | 1 | **-56** ✓ |
| 2 | High int | Proactive | 2.7s | 11 | 1 | -221 |
| 3 | High int | Reactive | 1.5s | 5 | 1 | **-118** ✓ |
| 4 | High fail | Aggressive | 2.4s | 13 | 2 | -153 |
| 5 | Fast forget | Proactive | 3.1s | 26 | 0 | -163 |

**Best performers marked with ✓**

---

## 🛠️ Technical Details

### **Video Specifications**:
- **Format**: MP4 (H.264)
- **Frame rate**: 10 fps
- **Resolution**: 1400×600 pixels
- **Duration**: 1.5-3.1 seconds per video
- **File size**: 90-183 KB per video

### **Simulation Mapping**:
- **2 simulation ticks** = 1 animation frame
- Each video shows 1 complete episode (start to serving)
- Real-time progression (not sped up or slowed down)

### **Kitchen Layout**:
```
┌─────────────────────────────────────────┐
│  [ONION]      [COUNTER]      [DISH]     │
│                                          │
│               [POT]                      │
│                                          │
│                             [SERVE]      │
└─────────────────────────────────────────┘
```

Agent path: ONION → POT → POT (wait) → DISH → SERVE

---

## 🔬 Research Implications

### **1. Context-Dependent Assistance**

These videos visually demonstrate that optimal assistant behavior is **highly context-dependent**:

- **Cooking (Video 1)**: Moderate reminders work fine
- **Surgery (Video 4)**: Aggressive reminders justified
- **Driving (Video 3)**: Minimal reactive interventions best

**Design Principle**: Assistants must infer context (c_int, c_fail) and adapt strategy dynamically.

---

### **2. The Interruption Cost Paradox**

Videos 1 vs. 2 show the paradox clearly:
- Same policy
- Same failures prevented
- 3.9× performance difference

**Implication**: "More helpful" ≠ "More useful" when interruptions are costly.

---

### **3. Memory as Control Variable**

Video 5 shows memory dynamics in action:
- Memory decays (watch right panel)
- Reminders boost memory
- Failures occur when memory drops too low

**Implication**: Memory is not a side effect - it's a first-class state variable that must be actively managed.

---

### **4. Individual Differences**

Video 5 (fast forgetting) requires 2× more reminders than others.

**Implication**: One-size-fits-all assistants will over-help slow forgetters and under-help fast forgetters. Must personalize to individual memory characteristics.

---

## 🎓 Educational Use

### **For Presentations**:

1. **Show Video 1** → "This is what proactive assistance looks like when interruptions are cheap"

2. **Show Video 2** → "Same policy, but interruptions are now expensive - performance collapses"

3. **Show Video 3** → "Reactive policy adapts by waiting until necessary - much better"

4. **Key message**: "The optimal assistant is context-dependent. There's no universal 'best' strategy."

### **For Research Discussions**:

- Use Videos 2 vs. 3 to discuss policy adaptation
- Use Video 5 to discuss memory dynamics and personalization
- Use Video 4 to discuss high-stakes domains (medical, safety)

### **For Design Reviews**:

- "Should our assistant behave like Video 1 (proactive) or Video 3 (reactive)?"
- "What's the c_int for our target users? What's the c_fail?"
- "Are we optimizing for the right cost structure?"

---

## 📝 How to Generate More Videos

### **Custom Configuration**:

```python
from procedure_assistant_sim import *
from visualize_kitchen import KitchenAnimator

# Define your parameters
params = SimulationParams(
    c_int=10.0,       # Your interruption cost
    c_fail_base=30.0, # Your failure cost
    lambda_forget=0.07 # Your forgetting rate
)

# Choose your policy
policy = ProactiveReminderPolicy(memory_threshold=0.3)

# Run one episode
result = run_simulation(policy, params, n_episodes=1, verbose=False)
history = result['histories'][0]

# Create video
animator = KitchenAnimator(
    history, params,
    title="My Custom Configuration"
)
animator.create_animation(
    output_path='videos/my_video.mp4',
    fps=10
)
```

### **Batch Generation**:

```bash
source venv/bin/activate
python visualize_kitchen.py  # Generates all 5 videos
```

---

## 📍 Video Locations

All videos are saved in:
```
/Users/arakawariku/Dropbox/Research/Antti/videos/
```

Files:
1. `video1_low_cost_proactive.mp4`
2. `video2_high_cost_proactive.mp4`
3. `video3_high_cost_reactive.mp4`
4. `video4_high_fail_cost.mp4`
5. `video5_fast_forgetting.mp4`

---

## 🎬 Conclusion

These videos transform abstract POMDP dynamics into concrete, interpretable visualizations. They reveal:

1. **The interruption cost paradox** (Videos 1-2)
2. **The value of adaptive policies** (Videos 2-3)
3. **Context-dependent optimization** (Videos 1, 3, 4)
4. **Memory dynamics** (Video 5)

By mapping procedural steps to kitchen movements and showing costs in real-time, these visualizations make the research findings tangible and memorable.

**Use them to communicate the key insight**: Optimal assistance is not about capability - it's about context-aware restraint.

---

**End of Video Guide**

For questions about the visualization code, see `visualize_kitchen.py`.
For questions about the simulation, see `procedure_assistant_sim.py` and documentation files.
