"""Re-evaluate v6 best checkpoints and report per-episode communication
action counts (mean ± std) across evaluation episodes.

For each (task, comm-cost c) cell of the commsweep_v6 sweep, load the best
human/assistant models, run N deterministic episodes, and record per-episode
counts of narrate / question / remind / confirm (plus failures, reward).

Output: data/results/emergent_comm_v6.json  (+ printed table)
"""
import sys, json, glob, os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions
from regime_definitions import build_params_asymmetric
from ma_procedure_assistant_sim import MAProcedureAssistantEnv
from train_ma_ippo import HumanGymWrapper, AssistantGymWrapper
from stable_baselines3 import PPO

N_EVAL = 200
TASKS = ['make_coffee','make_sandwich','make_cereal','make_tea',
         'cooking','latte_making','make_stencil']
CVALS = [0.5, 1.58, 5.0]


def eval_cell(task, c, n=N_EVAL):
    td = load_task_definitions()[task]
    p = build_params_asymmetric(td, c_fail_scale=15.0, c_nar=c, c_remind=c,
                                obs_regime='durable')
    env = MAProcedureAssistantEnv(p, td)
    cstr = f"{c:.4f}"
    mdir = PROJECT_ROOT / "models" / "ma_ippo" / task / f"asym_cn{cstr}_cr{cstr}_seed0"
    hm = PPO.load(str(mdir / "human_model_best"))
    am = PPO.load(str(mdir / "assistant_model_best"))
    wh = HumanGymWrapper(env, am); wa = AssistantGymWrapper(env, hm)
    rec = {k: [] for k in ['narrations','questions','reminds','confirms',
                           'failures','reward']}
    for _ in range(n):
        h_dict, a_dict = env.reset()
        h_obs = wh._convert_human_obs(h_dict); a_obs = wa._convert_assistant_obs(a_dict)
        R = 0.0; done = False
        while not done:
            ha, _ = hm.predict(h_obs, deterministic=True)
            aa, _ = am.predict(a_obs, deterministic=True)
            h_dict, a_dict, r, done, info = env.step(int(ha), int(aa))
            h_obs = wh._convert_human_obs(h_dict); a_obs = wa._convert_assistant_obs(a_dict)
            R += r
        s = env.ma_state
        rec['narrations'].append(s.total_narrations)
        rec['questions'].append(s.total_questions)
        rec['reminds'].append(s.total_reminds)
        rec['confirms'].append(s.total_confirms)
        rec['failures'].append(s.total_failures)
        rec['reward'].append(R)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in rec.items()}


def main():
    out = {}
    for t in TASKS:
        out[t] = {}
        for c in CVALS:
            cstr = f"{c:.4f}"
            mdir = PROJECT_ROOT / "models" / "ma_ippo" / t / f"asym_cn{cstr}_cr{cstr}_seed0"
            if not (mdir / "human_model_best.zip").exists():
                print(f"skip {t} c={c} (no model)"); continue
            res = eval_cell(t, c)
            out[t][str(c)] = res
            print(f"{t:13s} c={c:<4} | "
                  f"nar {res['narrations'][0]:5.2f}±{res['narrations'][1]:4.2f}  "
                  f"q {res['questions'][0]:5.2f}±{res['questions'][1]:4.2f}  "
                  f"rem {res['reminds'][0]:5.2f}±{res['reminds'][1]:4.2f}  "
                  f"con {res['confirms'][0]:5.2f}±{res['confirms'][1]:4.2f}  "
                  f"| fail {res['failures'][0]:4.2f}  r {res['reward'][0]:+6.2f}")
    op = PROJECT_ROOT / "data" / "results" / "emergent_comm_v6.json"
    json.dump(out, open(op, 'w'), indent=2)
    print(f"\nSaved: {op}")


if __name__ == '__main__':
    main()
