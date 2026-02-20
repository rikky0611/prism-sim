# Overnight Experiment Status

## ✅ What Was Done While You Slept

### 1. Task Updates (Sparse Action Space)
All 7 tasks updated with **conservative critical steps** (only 2-4 critical steps per task):

- **make_cereal**: Steps 3-4 critical (pour_cereal, pour_milk) → **4 actions**
- **make_coffee**: Step 5 critical (brew_coffee) → **3 actions**
- **make_tea**: Steps 3, 5 critical (heat_water, pour_water) → **4 actions**
- **make_sandwich**: Step 7 critical (prepare_sandwich) → **3 actions**
- **make_stencil**: Steps 2, 5, 8, 9 critical (exhaust, laser ops) → **6 actions**
- **cooking**: Steps 3, 6, 9, 11 critical (heat/safety) → **6 actions**
- **latte_making**: Steps 5, 9 critical (brew, steam) → **4 actions**

### 2. Parameter Updates
- **Baseline failure rate**: Increased from 0.3 to **0.6 (60%)** → Makes reminders **2× more beneficial**
- **Observation noise**: Disabled (0.0) → Agent sees true step
- **Off-timing penalty**: 0.5 → Penalizes reminding wrong steps

### 3. Training Status (21 models)

**Started**: Training all 3 regimes × 7 tasks = 21 models
- 200,000 timesteps per model
- Estimated runtime: **~6-7 hours**

**Check training progress**:
```bash
tail -100 /home/ec2-user/prism-sim/training_output.log
```

**Check if training completed**:
```bash
tail -50 /home/ec2-user/prism-sim/training_output.log | grep "TRAINING COMPLETE"
```

## 🔍 Morning Checklist

### Step 1: Verify Training Completed

```bash
cd /home/ec2-user/prism-sim
tail -100 training_output.log
```

Look for: `"✓ All models trained successfully!"` or count how many succeeded.

### Step 2: Run Evaluation

```bash
cd /home/ec2-user/prism-sim/src/experiments
python3 evaluate_cross_task_all_regimes.py --n-episodes 100
```

This will evaluate all 21 models against random baseline (~30 min runtime).

### Step 3: Generate Report

```bash
cd /home/ec2-user/prism-sim/src/visualization
python3 generate_cross_task_report.py
```

This creates a markdown report at: `results/cross_task_experiment_report.md`

### Step 4: Read Results

```bash
cat /home/ec2-user/prism-sim/results/cross_task_experiment_report.md
```

## 📊 Key Questions to Answer

1. **Did agent discover reminding?**
   - Check if RL models have **non-zero interventions** (not just silence)
   - Target: >50% of models should use reminders

2. **Did higher failure rate help?**
   - Compare improvements: Are they larger than previous experiments?
   - Previous: make_cereal had 0-5 interventions with 4+ failures
   - Expected: 2-4 interventions with <1 failure (very_high_stakes)

3. **Does sparse action space work?**
   - Check failures: Should be concentrated on critical steps only
   - Check interventions: Should align with critical steps (e.g., every 2-3 steps for make_cereal)

4. **Cross-task patterns?**
   - Do simple tasks (make_cereal, make_coffee) have lower improvements?
   - Do complex tasks (make_stencil, cooking) have higher improvements?
   - Do very_high_stakes models intervene more than moderate_low?

## 📁 Output Locations

- **Training log**: `/home/ec2-user/prism-sim/training_output.log`
- **Training summary**: `/home/ec2-user/prism-sim/data/results/cross_task_training_summary.json`
- **Models**: `/home/ec2-user/prism-sim/models/{regime}/{task}/final_model/`
- **Evaluation results**: `/home/ec2-user/prism-sim/data/results/cross_task_evaluation_results.json`
- **Final report**: `/home/ec2-user/prism-sim/results/cross_task_experiment_report.md`

## 🔧 If Training Failed

If some models failed to train, check:

```bash
# See which models failed
cat /home/ec2-user/prism-sim/data/results/cross_task_training_summary.json | grep -A 3 "failed"

# Check error details
grep "FAILED\|ERROR\|Exception" /home/ec2-user/prism-sim/training_output.log
```

You can retrain individual models:

```bash
cd /home/ec2-user/prism-sim/src/training
python3 train_single_task_multi_regime.py --task <task_name> --timesteps 200000
```

## 🎯 Expected Results (Hypothesis)

With f0_base=0.6 (60% baseline failure) and sparse action spaces:

**Very High Stakes (c_fail=30)**:
- Expected: 2-4 interventions per episode
- Expected: <1 failure per episode
- Expected: Non-zero interventions (agent tries reminding!)

**Balanced (c_fail=15)**:
- Expected: 1-3 interventions per episode
- Expected: 1-2 failures per episode

**Moderate Low (c_fail=10)**:
- Expected: 0-2 interventions per episode
- Expected: 2-3 failures per episode

**If results still show strategic silence** (0 interventions), then:
- Consider increasing f0_base further (0.7-0.8)
- Consider increasing off-timing penalty (1.0-2.0)
- Consider stronger reminders (δ_reminder=0.8)

Good luck with the results! 🚀
