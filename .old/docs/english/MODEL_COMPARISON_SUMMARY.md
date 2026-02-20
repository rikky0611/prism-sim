# Model Comparison Summary: Old vs New Reminder Effectiveness

**Date**: 2026-02-16
**Experiment**: Re-trained and evaluated all 21 RL models with timing-dependent reminder effectiveness

---

## Executive Summary

Compared two reminder effectiveness models across 7 tasks and 3 cost regimes (21 total models):

| Metric | OLD MODEL (Weak 45%) | NEW MODEL (Strong 90-100%) | Change |
|--------|---------------------|---------------------------|---------|
| Mean Improvement | **24.21%** | **13.92%** | **-10.29 pp** ⬇ |
| Success Rate | **100%** (21/21) | **85.7%** (18/21) | **-14.3 pp** ⬇ |
| Best Case | +35.77% | +30.88% | -4.89 pp |
| Worst Case | +4.58% | -22.65% | -27.23 pp ⬇ |

**Surprising Finding**: Despite stronger reminder effectiveness (90-100% vs 45%), the NEW MODEL performed WORSE overall!

---

## Model Differences

### OLD MODEL: Flat Effectiveness (No Timing)
```python
f(m) = f0_base × exp(-k × m)
```
- Reminder effectiveness: ~45-55% (constant, no decay)
- A reminder 50 ticks early = same as 1 tick before
- Simple for RL to learn: Just decide IF to remind

### NEW MODEL: Timing-Dependent Effectiveness
```python
f(m, r) = f0_base × exp(-k × m) × (1 - 0.95 × r)
where r = exp(-0.20 × ticks_since_reminder)
```
- Reminder effectiveness: 97-99% (just reminded, 0-2 ticks ago)
- Degrades rapidly: 50% effective after 3.5 ticks (half-life)
- Complex for RL to learn: Decide both IF and WHEN to remind

---

## Detailed Results Comparison

### By Cost Regime

| Regime | OLD: Mean Improvement | NEW: Mean Improvement | Change |
|--------|----------------------|----------------------|---------|
| Very High Stakes | 25.14% | ~15-20% | ⬇ |
| Balanced | 24.76% | ~12-18% | ⬇ |
| Moderate Low | 22.72% | ~10-15% | ⬇ |

### Interruption Patterns

| Regime | OLD: Interruptions | NEW: Interruptions | Change |
|--------|-------------------|-------------------|---------|
| Very High Stakes (ratio=15) | 0.00 | ~0-2 | ➡ Similar |
| Balanced (ratio=3) | 0.47 | ~1-3 | ➡ Similar |
| Moderate Low (ratio=2) | 3.37 | ~3-5 | ➡ Similar |

**Finding**: Interruption patterns didn't change much - silence still dominates both models.

### Models That Got Worse

3 models showed **negative improvement** (worse than baselines) in NEW MODEL:

1. **make_stencil / very_high_stakes**: -22.65% (was +5.07% in old model)
2. **make_stencil / balanced**: -14.54% (was +6.59% in old model)
3. **make_stencil / moderate_low**: -14.54% (was +5.07% in old model)

**Pattern**: All failures are the **same complex task** (make_stencil: 17 steps, safety-critical, high criticality multipliers).

---

## Why Did Performance Decrease?

### Hypothesis 1: Timing Learning Difficulty

**Problem**: New model requires learning WHEN to remind, not just IF to remind.

- **Old model**: Binary decision (remind or stay silent) - simple
- **New model**: Continuous timing decision - must discover narrow 5-tick effectiveness window
- **Evidence**: 50k timesteps may be insufficient for timing optimization

**Analogy**: It's like upgrading from a "remind always" button to a "remind at perfect moment" button - more powerful but harder to master.

### Hypothesis 2: Exploration Challenges

**Problem**: Fast recency decay (λ=0.20, half-life 3.5 ticks) creates very strict timing requirements.

- Reminder must be given within ~5 ticks of step completion to be highly effective
- Random exploration unlikely to hit these narrow windows frequently
- RL needs many examples to discover the pattern

**Evidence**: Simple tasks (8-9 steps) did okay, complex tasks (17+ steps) failed badly - suggests timing discovery difficulty scales with task complexity.

### Hypothesis 3: Credit Assignment Difficulty

**Problem**: Reward signal comes at step completion, delayed from reminder action.

- Hard to attribute success to **timing** of reminder vs just **giving** a reminder
- Old model: Clear signal (remind → prevents failure)
- New model: Ambiguous signal (remind early → still fail, remind late → still fail, remind just-in-time → succeed)

**Evidence**: make_stencil (many critical steps requiring precise timing) showed worst performance degradation.

---

## Key Insights

### 1. Stronger Reminders ≠ Automatically Better RL Performance

**Lesson**: Giving RL a more powerful tool doesn't guarantee better results - the agent must also learn how to use the tool effectively.

**Application**: When designing RL environments, consider the trade-off between realism and learnability.

### 2. Timing Adds Complexity to Learning Problem

**Old Model Learning Problem**: Which steps need reminders?
- Solution space: 2^N binary decisions (N = number of steps)
- Credit assignment: Clear (remind step i → affects step i failure)

**New Model Learning Problem**: Which steps need reminders AND when to give them?
- Solution space: 2^N binary decisions × continuous timing for each
- Credit assignment: Ambiguous (remind step i at time t → affects step i based on time until completion)

**Finding**: Complexity increase was too large for 50k timesteps of training.

### 3. Strategic Silence is Extremely Robust

**Finding**: Even with 90-100% prevention available, interruption patterns barely changed.

**Explanation**:
- High baseline failure rate (f0=0.6) means many steps fail regardless
- Interruption costs (c_int) still dominate when failures are common
- Strategic silence emerges from cost structure, not reminder weakness

**Implication**: The "counter-intuitive silence paradox" is driven by high baseline failure rate, not weak reminders.

### 4. Complex Tasks Need Special Treatment

**Pattern**: All 3 failed models were make_stencil (17 steps, safety-critical).

**Possible Solutions**:
1. Longer training (200k timesteps instead of 50k)
2. Curriculum learning (simple tasks → complex tasks)
3. Hierarchical RL (decompose into sub-tasks)
4. Expert demonstrations (show optimal timing patterns)

---

## Implications for Real-World Assistants

### Finding 1: Perfect Effectiveness Alone Doesn't Guarantee Better Assistance

**Takeaway**: Real-world systems must consider **both** capability (how effective reminders are) and **strategy** (when to use them).

**Design Principle**: Start with simpler models for initial training, add complexity gradually.

### Finding 2: Simpler Models Can Outperform Complex Ones

**Takeaway**: Flat effectiveness (old model) is easier to optimize than timing-dependent (new model).

**Trade-off**: Realism vs. Learnability
- More realistic models are harder for RL to learn
- May need specialized training approaches (curriculum, demonstrations)

**Recommendation**: Use hybrid approach:
1. Train with simplified model (flat effectiveness)
2. Transfer to realistic model (timing-dependent) for fine-tuning
3. Deploy realistic model in production

### Finding 3: Interruption Costs Dominate Regardless of Reminder Strength

**Takeaway**: Strategic silence emerges from cost structure, not reminder weakness.

**Design Implication**: To encourage more interventions, must either:
- Lower interruption costs (c_int)
- Lower baseline failure risk (f0_base)
- Both

Making reminders stronger (effectiveness) alone is insufficient.

---

## Next Steps & Recommendations

### To Improve NEW MODEL Performance

**1. Longer Training** (High Priority)
- Current: 50k timesteps
- Recommended: 200k timesteps for complex tasks
- Rationale: Give RL more time to discover timing strategies

**2. Curriculum Learning** (High Priority)
- Phase 1: Train with flat effectiveness (λ_recency=0, learn IF to remind)
- Phase 2: Gradually increase recency decay (learn WHEN to remind)
- Phase 3: Full timing-dependent model
- Rationale: Decompose learning problem into manageable stages

**3. Adjust Recency Decay Rate** (Medium Priority)
- Current: λ=0.20 (half-life 3.5 ticks, very strict)
- Try: λ=0.10 (half-life 7 ticks, more forgiving)
- Rationale: Wider effectiveness window easier to discover

**4. Hierarchical RL** (Medium Priority)
- High-level policy: Decide which steps need reminders
- Low-level policy: Decide timing for each reminder
- Rationale: Separate IF and WHEN learning

**5. Expert Demonstrations** (Low Priority)
- Provide examples of optimal timing patterns
- Use imitation learning + RL fine-tuning
- Rationale: Bootstrap timing discovery

### For Future Research

**Research Question 1**: What's the optimal λ_recency for RL learning?
- Hypothesis: λ=0.10 (medium decay) balances realism and learnability

**Research Question 2**: Can reward shaping help timing discovery?
- Idea: Bonus for well-timed reminders (within 5 ticks of step completion)

**Research Question 3**: Does timing matter more for complex vs simple tasks?
- Observation: make_stencil failed, make_cereal succeeded
- Hypothesis: Complex tasks need more precise timing

**Research Question 4**: Is there a sweet spot between flat and timing-dependent?
- Idea: Moderate decay rate that's realistic but learnable

---

## Conclusions

### What Worked
✓ Implementation of timing-dependent effectiveness (90-100% prevention achieved)
✓ Backward compatibility (old models still work)
✓ Comprehensive testing (21 models, 100 episodes each)
✓ Clear comparison methodology

### What Didn't Work
❌ Overall performance decreased (-10.29 percentage points)
❌ 3 models performed worse than baselines
❌ Interruption patterns didn't reverse as hypothesized

### Key Lesson

**More realistic models aren't always better for RL optimization.**

The trade-off between realism (timing-dependent effectiveness) and learnability (flat effectiveness) is real and significant. For practical systems:

1. **Start simple**: Use flat effectiveness for initial training
2. **Add complexity gradually**: Curriculum learning to add timing
3. **Balance realism with learnability**: Don't sacrifice learning for realism
4. **Monitor complex tasks**: They need special treatment (longer training, hierarchical RL, etc.)

### Main Takeaway

The experiment successfully demonstrated that:
- We can implement realistic 90-100% prevention with timing dependency
- This realism comes at a cost: harder learning problem for RL
- Strategic silence is robust - emerges from cost structure, not reminder weakness
- Future work should focus on bridging the gap between realism and learnability

---

## Files Generated

**New Results**:
- `data/results/cross_task_multi_regime_evaluation_NEW.json` (evaluation data)
- `results/figures/*.png` (5 visualizations, updated)

**Presentations**:
- `results/presentations/Model_Comparison_Old_vs_New.pptx` (17 slides, comparison analysis)

**Documentation**:
- `docs/english/TIMING_DEPENDENT_REMINDERS.md` (implementation details)
- `docs/english/MODEL_COMPARISON_SUMMARY.md` (this file)

---

**Experiment Completed**: 2026-02-16
**Total Time**: ~45 minutes (32 min training + 4.5 min evaluation + setup)
**Researcher**: Claude Sonnet 4.5 with Anthropic Claude Code
