# Cross-Task Multi-Regime Experiment Results

## Configuration

- **Tasks**: 7 (make_cereal, make_coffee, make_tea, make_sandwich, make_stencil, cooking, latte_making)
- **Regimes**: 3 (very_high_stakes, balanced, moderate_low)
- **Total Models**: 21
- **Training**: 200,000 timesteps per model
- **Baseline Failure Rate**: 60% (f0_base=0.6)
- **Action Space**: Sparse (only 2-4 critical steps per task)

## Summary Table

| Task | Regime | RL Reward | RL Interventions | RL Failures | Random Reward | Improvement |
|------|--------|-----------|------------------|-------------|---------------|-------------|
| make_cereal | very_high_stakes | -36.6 ± 21.0 | 0.00 | 4.88 | -75.4 | 51.4% |
| make_coffee | very_high_stakes | -11.0 ± 13.9 | 0.57 | 4.52 | -67.1 | 83.6% |
| make_tea | very_high_stakes | -33.3 ± 20.7 | 0.00 | 5.23 | -85.8 | 61.2% |
| make_sandwich | very_high_stakes | -6.7 ± 11.8 | 1.00 | 4.96 | -80.4 | 91.7% |
| make_stencil | very_high_stakes | -112.5 ± 26.6 | 67.33 | 8.73 | -191.6 | 41.3% |
| cooking | very_high_stakes | -72.6 ± 32.4 | 0.00 | 8.43 | -149.1 | 51.3% |
| latte_making | very_high_stakes | -37.2 ± 21.3 | 0.00 | 12.15 | -189.7 | 80.4% |
| make_cereal | balanced | -18.0 ± 9.5 | 0.00 | 4.68 | -71.2 | 74.7% |
| make_coffee | balanced | -4.0 ± 5.8 | 1.17 | 4.45 | -69.2 | 94.2% |
| make_tea | balanced | -16.5 ± 10.5 | 0.00 | 5.09 | -82.6 | 80.0% |
| make_sandwich | balanced | -9.4 ± 7.5 | 1.00 | 5.20 | -78.7 | 88.0% |
| make_stencil | balanced | -34.8 ± 15.3 | 2.04 | 10.04 | -175.3 | 80.2% |
| cooking | balanced | -35.7 ± 14.2 | 0.00 | 8.38 | -135.8 | 73.7% |
| latte_making | balanced | -16.5 ± 10.1 | 0.00 | 11.86 | -183.1 | 91.0% |
| make_cereal | moderate_low | -7.1 ± 6.3 | 1.41 | 4.20 | -73.9 | 90.4% |
| make_coffee | moderate_low | -3.8 ± 4.3 | 0.76 | 4.49 | -67.4 | 94.4% |
| make_tea | moderate_low | -12.0 ± 7.2 | 0.00 | 5.34 | -78.8 | 84.8% |
| make_sandwich | moderate_low | -3.3 ± 3.8 | 1.41 | 5.06 | -73.4 | 95.4% |
| make_stencil | moderate_low | -53.7 ± 14.8 | 22.50 | 10.19 | -167.1 | 67.9% |
| cooking | moderate_low | -24.1 ± 10.4 | 0.00 | 8.24 | -137.0 | 82.4% |
| latte_making | moderate_low | -12.3 ± 7.3 | 0.00 | 12.46 | -179.6 | 93.1% |


## Key Findings

- **Total Successful Evaluations**: 21/21
- **Best Performing Model**: make_sandwich / moderate_low (reward: -3.3)
- **Largest Improvement**: make_sandwich / moderate_low (+95.4%)
- **Overall Mean Improvement**: 78.6%
- **Median Improvement**: 82.4%

## Intervention Analysis

### Mean Interventions per Episode by Regime

- **very_high_stakes**: 9.84 (avg across 7 tasks)
- **balanced**: 0.60 (avg across 7 tasks)
- **moderate_low**: 3.73 (avg across 7 tasks)


## Failure Analysis

### Mean Failures per Episode by Regime

- **very_high_stakes**: 6.99 (avg across 7 tasks)
- **balanced**: 7.10 (avg across 7 tasks)
- **moderate_low**: 7.14 (avg across 7 tasks)


## Success Criteria Check

- **Models with non-zero interventions**: 10/21 (47.6%)
- **Models with >20% improvement**: 21/21
- **Models with <2 failures on average**: 0/21

## Conclusion

⚠ PARTIAL: Some models learned reminding, but not majority.

Generated on: 2026-02-17 17:12:38
