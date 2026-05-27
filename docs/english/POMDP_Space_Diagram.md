# Dec-POMDP Space Diagram: Multi-Agent Procedure Assistant

**System**: Cooperative Dec-POMDP with 2 agents (Human + Assistant)
**Task**: Sequential procedural tasks (e.g., make_cereal: N=8 steps, Nc=2 critical)
**Training**: IPPO (Independent PPO, alternating best-response)

---

## Figure 0: Dec-POMDP System Overview (Simplified)

> High-level relationships between the environment, human, and assistant.
> Focus on information asymmetry and action effects.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'clusterBkg': '#f5f5f5',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TB

  classDef stateNode  fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:3px
  classDef humanNode  fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef assistNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef rewardNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px

  %% ── ENVIRONMENT ──────────────────────────────────────────────────────────
  subgraph ENV["  Environment  "]
    STATE["sₜ  :  step · τ · memory · obs_noise"]:::stateNode
  end

  %% ── HUMAN AGENT ──────────────────────────────────────────────────────────
  subgraph HUMAN["  Human Agent  (MDP)  "]
    H_OBS["oᴴ  :  step · τ · memory"]:::humanNode
    H_ACT["Aᴴ  :  silent  /  narrate  /  question_j"]:::humanNode
    H_OBS --> H_ACT
  end

  %% ── ASSISTANT AGENT ──────────────────────────────────────────────────────
  subgraph ASST["  Assistant Agent  (POMDP)  "]
    A_BEL["bₜ  :  P(step = s)  Bayesian posterior over N+1 states"]:::assistNode
    A_ACT["Aᴬ  :  silent  /  confirm  /  remind_j"]:::assistNode
    A_BEL --> A_ACT
  end

  %% ── SHARED REWARD ────────────────────────────────────────────────────────
  REWARD["Shared Reward  rₜ  :  −c_fail  −c_int  −c_nar  −c_q  +R_complete"]:::rewardNode

  %% ── OBSERVATIONS ─────────────────────────────────────────────────────────
  STATE --> H_OBS
  STATE -. "noisy channel  p = obs_noise" .-> A_BEL

  %% ── MUTUAL LAST-ACTION VISIBILITY ────────────────────────────────────────
  H_ACT -. "last action visible in oᴬ" .-> A_ACT
  A_ACT -. "last action visible in oᴴ" .-> H_OBS

  %% ── NARRATION: KEY INFO BRIDGE ───────────────────────────────────────────
  H_ACT == "narrate  →  belief hard-reset to true step" ==> A_BEL

  %% ── EFFECTS ON WORLD ─────────────────────────────────────────────────────
  H_ACT -->|"question_j: memory up / narrate: noise down"| STATE
  A_ACT -->|"remind_j: memory up"| STATE

  %% ── REWARD ───────────────────────────────────────────────────────────────
  STATE -->|"step failure"| REWARD
  H_ACT -->|"c_nar · c_q"| REWARD
  A_ACT -->|"c_int · c_confirm"| REWARD
```

---

## Figure 1: Experiment Parameter Summary

> Key parameter values used across all experiments.
> Left: fixed environment dynamics. Center: cost regimes (main experimental variable). Right: noise conditions.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'clusterBkg': '#f5f5f5',
    'fontFamily': 'monospace'
  }
}}%%

flowchart LR

  classDef dynNode    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef maNode     fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef lowNode    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef balNode    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef highNode   fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef sharedNode fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef noiseNode  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef headerNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:1px

  %% ── ENVIRONMENT DYNAMICS (fixed) ─────────────────────────────────────────
  subgraph DYN["  Environment Dynamics  (fixed across all experiments)  "]
    direction TB
    D1["f₀ = 0.6  ·  k = 3.0"]:::dynNode
    D2["λ_forget = 0.03  (half-life 23 ticks)"]:::dynNode
    D3["Δ_reminder = 0.8  ·  Δ_q = 0.4"]:::maNode
    D4["step duration: μ = 30  ·  σ = 10  ticks"]:::dynNode
    D5["c_int = 1.0  ·  c_off_timing = 3.0"]:::sharedNode
    D6["c_q = 0.5  ·  c_confirm = 1.0"]:::maNode
    D7["R_complete = 10.0  (task completion bonus)"]:::dynNode
  end

  %% ── COST REGIMES (main experimental variable) ────────────────────────────
  subgraph REGIMES["  Cost Regimes  "]
    direction TB

    subgraph REG_L["  extremely_low  "]
      L1["c_fail_scale = 2.0"]:::lowNode
    end

    subgraph REG_B["  balanced  "]
      B1["c_fail_scale = 15.0"]:::balNode
    end

    subgraph REG_H["  extremely_high  "]
      H1["c_fail_scale = 50.0"]:::highNode
    end

    NOTE["c_fail_per_step = c_fail_scale × criticality"]:::sharedNode
  end

  %% ── NOISE CONDITIONS ─────────────────────────────────────────────────────
  subgraph NOISE["  Noise Conditions  "]
    direction TB

    subgraph N02["  noise02  (low)  "]
      NO1["obs_noise = 0.20"]:::noiseNode
      NO2["obs_noise_min = 0.02"]:::noiseNode
    end

    subgraph N05["  noise05  (high)  "]
      NO3["obs_noise = 0.50"]:::noiseNode
      NO4["obs_noise_min = 0.05"]:::noiseNode
    end

    NO5["λ_noise_recover = 0.10  (shared)"]:::noiseNode
  end
```

### Tasks Used in Experiments

| Task | Steps | Critical | base_fail_cost | Domain | Rollout Patterns |
|------|------:|--------:|---------------:|--------|:----------------:|
| make_cereal | 8 | 2 | 10.0 | cooking | 6 |
| make_coffee | 8 | 1 | 12.0 | cooking | — |
| make_tea | 9 | 2 | 12.0 | cooking | — |
| make_sandwich | 9 | 1 | 15.0 | cooking | — |
| cooking | 14 | 4 | 20.0 | cooking | — |
| make_stencil | 17 | 4 | 30.0 | crafting | — |
| latte_making | 20 | 2 | 25.0 | technical | 12 |

### PPO Hyperparameters & Training Configuration

| Hyperparameter | Value | Training Config | Standard | Noise Comparison |
|----------------|------:|----------------|--------:|----------------:|
| learning_rate | 3e-4 | n_rounds | 15 | 8 |
| n_steps (rollout) | 2048 | steps_per_round | 50,000 | 20,000 |
| batch_size | 64 | total steps/agent | 750,000 | 160,000 |
| n_epochs | 10 | eval episodes | 200 | 200 |
| gamma | 0.99 | training strategy | IPPO alternating | IPPO alternating |
| clip_range | 0.2 | | | |
| ent_coef | 0.01 | | | |

---

## Figure 1 (detailed): Dec-POMDP System Overview

> Overall architecture: hidden latent state, both agent observation/action/policy spaces,
> information flows, and shared reward decomposition.
> Dashed arrows = noisy or inferred channels. Solid arrows = perfect channels.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'primaryBorderColor': '#000000',
    'lineColor': '#000000',
    'secondaryColor': '#ffffff',
    'tertiaryColor': '#ffffff',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'nodeBorder': '#000000',
    'clusterBkg': '#f5f5f5',
    'titleColor': '#000000',
    'edgeLabelBackground': '#ffffff',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TB

  classDef hiddenState fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef humanNode  fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef assistNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef rewardNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef obsBox     fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef actionBox  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef policyBox  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef beliefBox  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef dynBox     fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px

  %% ── LATENT STATE (top) ───────────────────────────────────────────────────
  subgraph LS["  LATENT STATE  sₜ  (hidden from assistant)  "]
    direction LR
    S1["current_step  ∈  {0..N-1, done}"]:::hiddenState
    S2["τ  :  elapsed ticks in current step"]:::hiddenState
    S3["memory[N]  ∈  [0, 2]\nλ_forget=0.03  |  +Δ_q=0.4  |  +Δ_rem=0.8"]:::hiddenState
    S4["obs_noise  ∈  [noise_min, baseline]\ndrops on narrate  →  exp recovery  λ=0.10"]:::hiddenState
    S1 --> S2 --> S3 --> S4
  end

  %% ── HUMAN AGENT ──────────────────────────────────────────────────────────
  subgraph HA["  HUMAN AGENT  (full-info MDP)  "]
    direction LR
    H_OBS["oᴴ  —  Observation  (dim = 5)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\ncurrent_identity  (perfect)\nτ  (exact elapsed ticks)\nmemory[current_step]  (exact)\nassistant_last_action\nobs_noise_state"]:::obsBox
    H_POL["πᴴ  —  PPO Policy\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMLP  64 × 64  tanh\nshared actor-critic"]:::policyBox
    H_ACT["Aᴴ  —  Action Space  (size = 2 + Nc)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n0 :  silent\n1 :  narrate  →  drops obs_noise to min\n2+j :  question_j  →  memory[crit_j] += Δ_q"]:::actionBox
    H_OBS --> H_POL --> H_ACT
  end

  %% ── ASSISTANT AGENT ──────────────────────────────────────────────────────
  subgraph AA["  ASSISTANT AGENT  (POMDP, belief-based)  "]
    direction LR
    A_BEL["bₜ  —  Belief State\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nstep_belief[N+1]  :  P(step = s)\nupdated via Bayesian filter\n+ identity transition matrix T"]:::beliefBox
    A_OBS["oᴬ  —  Observation  (dim = N+1+Nc+1)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nstep_belief[N+1]  (own posterior)\nmemory_estimate_critical[Nc]\nhuman_last_action"]:::obsBox
    A_POL["πᴬ  —  PPO Policy\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nMLP  64 × 64  tanh\noperates on belief, not ground truth"]:::policyBox
    A_ACT["Aᴬ  —  Action Space  (size = 2 + Nc)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n0 :  silent\n1 :  confirm  →  costs c_confirm\n2+j :  remind_j  →  memory[crit_j] += Δ_rem\n        +c_off_timing if wrong step"]:::actionBox
    A_BEL --> A_OBS --> A_POL --> A_ACT
  end

  %% ── SHARED REWARD (bottom) ────────────────────────────────────────────────
  subgraph RW["  SHARED REWARD  rₜ  (fully cooperative)  "]
    direction LR
    R1["-c_fail × criticality\n  step failure"]:::rewardNode
    R2["-c_int  (remind)\n-c_confirm  (confirm)"]:::rewardNode
    R3["-c_nar  (narrate)\n-c_q  (question)"]:::rewardNode
    R4["-c_off_timing\n  wrong-step remind"]:::rewardNode
    R5["+R_complete\n  task done bonus"]:::rewardNode
  end

  %% ── FAILURE DYNAMICS ─────────────────────────────────────────────────────
  DYN["f(m, r) = f₀ · exp(−k·m) · (1 − ε·r)\nf₀=0.6  k=3.0  ε=0.95\nstep completion: discrete hazard model"]:::dynBox

  %% ── INFORMATION FLOWS ────────────────────────────────────────────────────
  LS -- "oᴴ: perfect identity, τ, memory, noise\n(perfect channel)" --> H_OBS
  LS -. "noisy obs: obs_noise corrupts step_id\n(channel noise = obs_noise_state)" .-> A_BEL
  H_ACT == "narrate  →  belief hard-reset\nto point mass at true_step" ==> A_BEL
  H_ACT -- "question_j  →  memory[j] += Δ_q\nnarrate  →  obs_noise drops" --> LS
  A_ACT -- "remind_j  →  memory[j] += Δ_rem" --> LS
  H_ACT -. "human_last_action  ∈  oᴬ" .-> A_OBS
  A_ACT -. "assistant_last_action  ∈  oᴴ" .-> H_OBS
  LS --> DYN --> R1
  H_ACT --> R3
  A_ACT --> R2
  A_ACT --> R4
  LS --> R5
  RW -- "shared reward" --> H_POL
  RW -- "shared reward" --> A_POL

  %% ── IPPO TRAINING ────────────────────────────────────────────────────────
  IPPO["IPPO Training\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\nRound odd :  fix assistant, train human\nRound even :  fix human, train assistant\n(alternate until convergence)"]:::policyBox
  H_POL --- IPPO --- A_POL
```

---

## Figure 2: Agent Space Comparison

> Side-by-side detailed view of Human vs. Assistant.
> Visual contrast between **perfect-info** observation and **belief-based / inferred** observation.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'nodeBorder': '#000000',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TB

  classDef humanHeader  fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px,font-size:14px
  classDef assistHeader fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px,font-size:14px
  classDef humanObs  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef assistObs fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef humanAct  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef assistAct fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef humanPol  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef assistPol fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef perfectTag fill:#ffffff,stroke:#000000,color:#000000,font-size:11px,stroke-width:1px
  classDef inferTag   fill:#ffffff,stroke:#000000,color:#000000,font-size:11px,stroke-width:1px
  classDef noisyTag   fill:#ffffff,stroke:#000000,color:#000000,font-size:11px,stroke-width:1px
  classDef asymNote   fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px

  %% ── HEADERS ──────────────────────────────────────────────────────────────
  HH["HUMAN AGENT\nfull-info MDP"]:::humanHeader
  AH["ASSISTANT AGENT\nPOMDP  (belief-based)"]:::assistHeader

  %% ── HUMAN OBSERVATION ────────────────────────────────────────────────────
  subgraph HOB["Observation Space  oᴴ  |  dim = 5"]
    direction TB
    H1["current_identity  :  which step is active  (0..N-1)"]:::humanObs
    H1T["PERFECT — exact ground truth"]:::perfectTag
    H2["τ  :  elapsed ticks in current step  (exact)"]:::humanObs
    H2T["PERFECT — exact count"]:::perfectTag
    H3["memory[current_step]  ∈ [0, 2]  (own long-term memory)"]:::humanObs
    H3T["PERFECT — own memory, exact"]:::perfectTag
    H4["assistant_last_action  ∈  {0..2+Nc-1}"]:::humanObs
    H5["obs_noise_state  ∈  [noise_min, baseline]"]:::humanObs
  end

  %% ── ASSISTANT OBSERVATION ────────────────────────────────────────────────
  subgraph AOB["Observation Space  oᴬ  |  dim = N+1 + Nc + 1"]
    direction TB
    A1["step_belief[N+1]  :  P(human on step s)  for s ∈ {0..N}"]:::assistObs
    A1T["INFERRED — Bayesian posterior, not ground truth"]:::inferTag
    A2["memory_estimate_critical[Nc]\nassistant's running estimate of critical-step memory\n(built from interaction history only — no ground truth access)"]:::assistObs
    A2T["INFERRED — shadow model from interactions"]:::inferTag
    A3["human_last_action  ∈  {0..2+Nc-1}"]:::assistObs
    A3T["OBSERVED — perfect channel"]:::perfectTag
  end

  %% ── HUMAN ACTIONS ────────────────────────────────────────────────────────
  subgraph HAC["Action Space  Aᴴ  |  |A| = 2 + Nc"]
    direction LR
    HA0["0 : silent\nno cost, no effect"]:::humanAct
    HA1["1 : narrate\ncost c_nar\nobs_noise → noise_min  (hard drop)\nassistant belief → point mass at true_step"]:::humanAct
    HA2["2+j : question_j  (j ∈ 0..Nc-1)\ncost c_q\nmemory[crit_j] += Δ_q = 0.4\nvisible to assistant via human_last_action"]:::humanAct
  end

  %% ── ASSISTANT ACTIONS ────────────────────────────────────────────────────
  subgraph AAC["Action Space  Aᴬ  |  |A| = 2 + Nc"]
    direction LR
    AA0["0 : silent\nno cost, no effect"]:::assistAct
    AA1["1 : confirm\ncost c_confirm\nno memory effect"]:::assistAct
    AA2["2+j : remind_j  (j ∈ 0..Nc-1)\ncost c_int\nmemory[crit_j] += Δ_rem = 0.8\n+ c_off_timing if wrong step"]:::assistAct
  end

  %% ── POLICIES ─────────────────────────────────────────────────────────────
  subgraph HPL["Policy  πᴴ"]
    direction TB
    HP1["Input :  oᴴ ∈ ℝ⁵"]:::humanPol
    HP2["PPO MLP  64 × 64  tanh\nshared actor-critic"]:::humanPol
    HP3["Output :  π(aᴴ | oᴴ)\nDiscrete(2+Nc)  softmax"]:::humanPol
    HP1 --> HP2 --> HP3
  end

  subgraph APL["Policy  πᴬ"]
    direction TB
    AP1["Input :  oᴬ ∈ ℝᴺ⁺¹⁺ᴺᶜ⁺¹"]:::assistPol
    AP2["PPO MLP  64 × 64  tanh\nshared actor-critic"]:::assistPol
    AP3["Output :  π(aᴬ | oᴬ)\nDiscrete(2+Nc)  softmax"]:::assistPol
    AP1 --> AP2 --> AP3
  end

  %% ── KEY ASYMMETRY ────────────────────────────────────────────────────────
  ASYM["KEY ASYMMETRY\nHuman sees exact ground truth (step identity, τ, own memory).\nAssistant sees only its Bayesian posterior belief and\nits own inferred shadow of memory — never ground truth.\nNarration is the sole perfect information channel from human to assistant."]:::asymNote

  %% ── CONNECTIONS ──────────────────────────────────────────────────────────
  HH --> HOB --> HAC --> HPL --> ASYM
  AH --> AOB --> AAC --> APL --> ASYM
```

---

## Figure 3: Assistant Belief Update Mechanism

> Per-tick belief update for the assistant: three paths depending on whether the human narrates.
> Memory estimate update runs in parallel every tick.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'nodeBorder': '#000000',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TD

  classDef tickStart  fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef pathA      fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef pathB      fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef pathC      fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef formula    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px,font-family:monospace
  classDef memPath    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef outputNode fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef decision   fill:#ffffff,stroke:#000000,color:#000000,stroke-width:2px
  classDef noiseNode  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef noteBox    fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px,stroke-dasharray:4 4

  %% ── TICK START ───────────────────────────────────────────────────────────
  TICK["TICK  t  BEGINS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nbₜ₋₁  :  step_belief from previous tick\nmem_est_crit  :  memory estimate for Nc critical steps\nhuman_action aᴴ,  assistant_action aᴬ  (simultaneous)"]:::tickStart

  %% ── SAMPLE NOISY OBS ─────────────────────────────────────────────────────
  NOBS["SAMPLE NOISY STEP OBSERVATION\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nobs_identity  ~  Noisy(true_step,  p_noise)\nwhere  p_noise = obs_noise_state\nwith prob p:  add Gaussian noise  N(0,1)  to step id"]:::pathA

  %% ── DECISION: narration? ─────────────────────────────────────────────────
  NARQ{"human_action\n==  narrate  ?"}:::decision

  %% ── PATH B: NARRATION HARD RESET ─────────────────────────────────────────
  subgraph PB["PATH B  —  Narration  (hard reset, perfect info)"]
    direction TB
    B1["Point-mass reset\nbₜ[s] = 1.0   if  s == true_step\nbₜ[s] = 0.0   otherwise"]:::pathB
    B2["Noise hard drop\nobs_noise_state  →  obs_noise_min\nticks_since_narration  =  0"]:::pathB
    B1 --> B2
  end

  %% ── PATH A: PRIOR TRANSITION ─────────────────────────────────────────────
  subgraph PA["PATH A  —  Prior Transition Update"]
    direction TB
    A1["Step-completion hazard:\nh_prior = 1 / step_mean_duration"]:::pathA
    A2["b̃ₜ  =  (1 − h_prior) · bₜ₋₁\n      +  h_prior · (Tᵀ · bₜ₋₁)\nT  =  identity transition matrix\n(averaged over task rollout patterns)"]:::formula
    A3["Normalize:   b̃ₜ  ←  b̃ₜ / ‖b̃ₜ‖₁"]:::pathA
    A1 --> A2 --> A3
  end

  %% ── PATH C: BAYESIAN LIKELIHOOD ──────────────────────────────────────────
  subgraph PC["PATH C  —  Bayesian Likelihood Update"]
    direction TB
    C1["Likelihood  (per hypothesis  s):\nℓ[s]  =  (1−p) · 𝟙(s == obs_identity)\n       +  p · N(obs_identity − s ;  0, 1)\nwhere  p = obs_noise_state"]:::formula
    C2["Posterior update:\nbₜ  ∝  b̃ₜ  ·  ℓ"]:::pathC
    C3["Normalize:   bₜ  ←  bₜ / ‖bₜ‖₁\n(fallback: uniform distribution if all-zero)"]:::pathC
    C1 --> C2 --> C3
  end

  %% ── NOISE RECOVERY (when no narration) ───────────────────────────────────
  NREC["OBSERVATION NOISE RECOVERY\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nnoise(t)  =  baseline − (baseline − min) · exp(−λ · t)\nλ_noise_recover = 0.10   (half-life ≈ 7 ticks)\nrecovery toward baseline; narration resets t = 0"]:::noiseNode

  %% ── MEMORY ESTIMATE UPDATE (parallel) ───────────────────────────────────
  subgraph PM["MEMORY ESTIMATE UPDATE  (parallel, runs every tick)"]
    direction LR
    M1["Decay\nmem_est  *=  (1 − λ_forget)\nλ_forget = 0.03"]:::memPath
    M2["Human question_j observed?\nmem_est[j]  +=  Δ_q = 0.4"]:::memPath
    M3["Own remind_j action?\nmem_est[j]  +=  Δ_rem = 0.8"]:::memPath
    M4["Clip\nmem_est  ←  clip(mem_est, 0, 2)"]:::memPath
    M1 --> M2 --> M3 --> M4
  end

  %% ── OUTPUTS ──────────────────────────────────────────────────────────────
  OUT["UPDATED BELIEF  bₜ\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nstep_belief[N+1]  :  posterior over step identities\n→  used as input to assistant PPO policy at tick t+1"]:::outputNode
  MOUT["UPDATED MEMORY ESTIMATE\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nmem_est_crit[Nc]\nassistant's shadow model of human critical-step memory"]:::outputNode

  %% ── CONCENTRATION NOTE ───────────────────────────────────────────────────
  NOTE["CONCENTRATION EFFECT\nLow obs_noise (post-narration)\n  →  ℓ sharply peaked  →  belief concentrates fast\nHigh obs_noise (baseline ~0.2)\n  →  ℓ spread out       →  belief diffuses slowly\nNarration bypasses this entirely (hard reset)"]:::noteBox

  %% ── EDGES ────────────────────────────────────────────────────────────────
  TICK --> NOBS
  NOBS --> NARQ

  NARQ -- "YES" --> PB
  NARQ -- "NO"  --> PA
  PA   --> PC
  NARQ -- "NO"  --> NREC

  PB --> OUT
  PC --> OUT
  TICK --> PM
  PM  --> MOUT

  NOTE -.-> PC
  NOTE -.-> PB
```

---

## Summary Table

| | Human Agent | Assistant Agent |
|---|---|---|
| **Observability** | Full information (MDP) | Partial (POMDP) |
| **Obs space dim** | 5 | N+1+Nc+1 |
| **Key obs** | Perfect step identity, τ, own memory | Bayesian belief, inferred mem_est |
| **Action space** | {silent, narrate, question_j} | {silent, confirm, remind_j} |
| **Action size** | 2+Nc | 2+Nc |
| **Belief update** | No belief needed | Bayesian filter + transition prior |
| **Memory access** | Exact (own memory) | Shadow estimate from interactions |
| **Info channel** | narrate → resets assistant belief | — |
| **Policy** | PPO MLP 64×64 | PPO MLP 64×64 |
| **Reward** | Shared cooperative rₜ | Shared cooperative rₜ |

**Parameters** (MASimulationParams defaults):
`f₀=0.6`, `k=3.0`, `λ_forget=0.03`, `Δ_reminder=0.8`, `Δ_q=0.4`, `λ_noise_recover=0.10`

---

## Figure 4: IPPO Training Procedure

> Alternating best-response loop used to train both agents jointly.
> One "round" = train human (assistant fixed) + train assistant (human fixed) + joint eval.
> 15 rounds × 3 cost regimes per task.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'clusterBkg': '#f5f5f5',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TD

  classDef bold fill:#ffffff,stroke:#000000,color:#000000,font-weight:bold,stroke-width:2px
  classDef box  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:1px
  classDef dec  fill:#ffffff,stroke:#000000,color:#000000,stroke-width:2px

  INIT["Initialize  π_H  and  π_A  with random weights"]:::bold
  REGIME["For each cost regime:  extremely_low  /  balanced  /  extremely_high"]:::box

  TH["Train  π_H  for 50,000 steps"]:::box
  TH_W["HumanGymWrapper  —  π_A  is frozen"]:::box

  TA["Train  π_A  for 50,000 steps"]:::box
  TA_W["AssistantGymWrapper  —  π_H  is frozen"]:::box

  EVAL["Joint Evaluation  —  50 episodes  (greedy)"]:::box
  EREWARD["Compute  mean_reward  and  mean_failures"]:::box

  BEST{"reward  >  best_reward?"}:::dec
  SAVE["Save checkpoint  to  models/ma_ippo/{task}/{regime}/"]:::box
  NEXT["Next round  (up to 15 rounds total)"]:::box

  FINAL["Load best checkpoint  →  Final Eval  (200 episodes)"]:::bold

  INIT --> REGIME --> TH --> TH_W --> TA --> TA_W --> EVAL --> EREWARD --> BEST
  BEST -- "yes" --> SAVE --> NEXT
  BEST -- "no" --> NEXT
  NEXT -.->|"repeat"| TH
  NEXT -->|"round 15 done"| FINAL
```

### PPO Configuration (both agents, shared)

| lr | n_steps | batch_size | n_epochs | γ | λ (GAE) | clip | ent_coef |
|----|---------|-----------|---------|---|---------|------|---------|
| 3e-4 | 2048 | 64 | 10 | 0.99 | 0.95 | 0.2 | 0.01 |

Policy network: MLP 64 × 64, tanh activation, shared actor-critic.

---

## Figure 5: Task Step Graphs

### make_cereal  (8 steps, 2 critical)

> Critical steps (★) colored red. Arrows show all possible step-to-step transitions across all 6 rollout patterns (18 unique edges).

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'fontFamily': 'monospace'
  }
}}%%

flowchart LR

  classDef crit fill:#ff6b6b,stroke:#cc0000,color:#ffffff,font-weight:bold,stroke-width:2px
  classDef norm fill:#ffffff,stroke:#333333,color:#000000,stroke-width:1px

  S0["0  get_bowl"]:::norm
  S1["1  get_cereal"]:::norm
  S2["2  get_milk"]:::norm
  S3["3  pour_cereal  ★"]:::crit
  S4["4  pour_milk  ★"]:::crit
  S5["5  get_spoon"]:::norm
  S6["6  return_items"]:::norm
  S7["7  serve_cereal"]:::norm

  S0 --> S1 & S2 & S5
  S1 --> S2 & S3 & S4
  S2 --> S1 & S3 & S4
  S3 --> S4 & S5
  S4 --> S3 & S5 & S6
  S5 --> S1 & S2 & S6
  S6 --> S7
```

---

### latte_making  (20 steps, 2 critical)

> Critical steps (★) colored red. Steps grouped into 3 phases. Arrows show all possible step-to-step transitions across all 12 rollout patterns (37 unique edges), including cross-phase and back-edges.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'background': '#ffffff',
    'mainBkg': '#ffffff',
    'clusterBkg': '#f5f5f5',
    'fontFamily': 'monospace'
  }
}}%%

flowchart TB

  classDef crit fill:#ff6b6b,stroke:#cc0000,color:#ffffff,font-weight:bold,stroke-width:2px
  classDef norm fill:#ffffff,stroke:#333333,color:#000000,stroke-width:1px

  subgraph PH1["  Phase 1: Espresso Setup  (steps 0–6)  "]
    direction LR
    S0["0  gather"]:::norm
    S1["1  grind_beans"]:::norm
    S2["2  prepare_portafilter"]:::norm
    S3["3  attach_portafilter"]:::norm
    S4["4  place_cup"]:::norm
    S5["5  brew_coffee  ★"]:::crit
    S6["6  remove_portafilter"]:::norm
    S0 --> S1 & S2
    S1 --> S2 & S3
    S2 --> S1 & S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
  end

  subgraph PH2["  Phase 2: Milk Preparation  (steps 7–12)  "]
    direction LR
    S7["7  pour_milk_pitcher"]:::norm
    S8["8  position_pitcher"]:::norm
    S9["9  steam_milk  ★"]:::crit
    S10["10  texture_milk"]:::norm
    S11["11  tap_pitcher"]:::norm
    S12["12  swirl_milk"]:::norm
    S7 --> S8
    S8 --> S9
    S9 --> S10
    S10 --> S11 & S12
    S11 --> S12
    S12 --> S11
  end

  subgraph PH3["  Phase 3: Assembly and Finish  (steps 13–19)  "]
    direction LR
    S13["13  pour_slowly"]:::norm
    S14["14  create_pattern"]:::norm
    S15["15  finish_pour"]:::norm
    S16["16  wipe_cup"]:::norm
    S17["17  clean_wand"]:::norm
    S18["18  clean_portafilter"]:::norm
    S19["19  serve_latte"]:::norm
    S13 --> S14
    S14 --> S15
    S15 --> S16
    S16 --> S17 & S18
    S17 --> S18 & S19
    S18 --> S17 & S19
  end

  %% Phase 1 → Phase 2 (forward cross-phase)
  S4 --> S7
  S5 --> S7 & S9
  S6 --> S7 & S9 & S10

  %% Phase 2 → Phase 1 (back-edges)
  S8 --> S5
  S9 --> S6

  %% Phase 2 → Phase 3 (forward cross-phase)
  S11 --> S13
  S12 --> S13
```

---

## Figure 6: Experiment Results

### make_cereal Results

| Regime | c_fail_scale | Best Baseline | MA-IPPO Final | Failure Rate | Learned Strategy |
|--------|-------------|:-------------:|:-------------:|:------------:|-----------------|
| extremely_low | 2.0 | 7.57 (silent) | **7.08** | 0.74 | Strategic silence (questions = 0) |
| balanced | 15.0 | -7.33 (silent) | **3.84** | 0.15 | Selective questions (≈6/episode) |
| extremely_high | 50.0 | -52.25 (silent) | **-1.49** | 0.16 | Active questions (≈6/episode) |

> Baseline: `both_silent` (human and assistant both always silent).
> Failures reduced 8× in balanced, 78× improvement vs baseline in extremely_high.

### latte_making Results

| Regime | c_fail_scale | Best Baseline | MA-IPPO Final | Failure Rate | Learned Strategy |
|--------|-------------|:-------------:|:-------------:|:------------:|-----------------|
| extremely_low | 2.0 | 7.45 (silent) | -1.47 | 0.24 | Strategic silence (questions ≈ 0) |
| balanced | 15.0 | -7.47 (silent) | -16.03 | 0.77 | Mostly silent (questions ≈ 0.1) |
| extremely_high | 50.0 | -49.25 (silent) | -48.25 | 1.17 | Silent — convergence difficulty |

> latte_making is significantly harder (20 steps vs 8). The agent struggles especially in
> balanced/extremely_high regimes. Failure rate remains high due to task complexity.

### Regime × Task Summary

| | make_cereal | latte_making |
|---|---|---|
| **Best regime for learning** | extremely_high (+97% vs baseline) | extremely_low (best relative) |
| **Strategy: extremely_low** | Strategic silence | Strategic silence |
| **Strategy: balanced** | Selective questions | Mostly silent |
| **Strategy: extremely_high** | Active questioning | Convergence difficulty |
| **Narrations learned** | 0 across all regimes | 0 across all regimes |
