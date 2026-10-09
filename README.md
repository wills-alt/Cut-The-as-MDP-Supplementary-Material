# Stealthy intrusion defense against MDP-based attackers

Code and data for: W. Kouam and S. Rass, *Game-theoretic stealthy intrusion defense against MDP-based attackers* (under review).

An attacker moves through an acyclic attack graph, modeled as a Markov decision process. The defender protects `h` nodes. Three attacker regimes are solved:

- **Stackelberg**: the attacker sees the deployment (MILP).
- **Blind**: the attacker knows nothing and assumes every node is protected with the same probability.
- **Dirichlet**: the attacker has a private random belief on where the protection is (MILP on sampled beliefs).

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The paper uses `SOLVER = "CBC"` to use the free CBC solver; using Gurobi (`SOLVER = "GRB"`, licence required) gives the same results.

## Reproducing the results

All outputs go to `csv_results/`.

| Command | Output | Paper |
|---|---|---|
| `python run_all.py` | results of the three regimes, graphs (JSON), `run_info.json` | Figs. 3-5, main tables |
| `python Sensitivity.py` | sensitivity to the belief parameters, convergence in K | sensitivity tables |
| `python Baselines.py` | betweenness, minimum cut and greedy baselines | baselines table |
| `python CTR_2023_comp.py` | comparison with Cut-The-Rope 2023 | comparison table |
| `python Assumption_1_exhibit.py` | numerical exhibit of Assumption 1 | Assumption 1 table |
| `python make_tables.py` | all LaTeX tables (`tables.tex`) |  |
| `python plot_results.py` | figures | Figs. 3-5 |


**Settings:** attack rate 2, defense rate 1 (exponential defender), budgets h = 1 to 5, uniform start on all non-target nodes. Dirichlet: lambda = 0.5, M = 10, 500 training beliefs (seed 1), 2000 test beliefs (seed 2). Random baseline: all deployments enumerated. The full configuration of each run is saved in `csv_results/run_info.json`.

**Graphs:** MARA (9 nodes), MiR100 (16 nodes), Unguard (33 nodes), defined in `GRAPHS` and exported to `csv_results/graphs/`.

## Files and functions

**`Attack_Graph_Implementation.py`**: the model.
- `AttackGraph`: graph, state space `(node, steps)`, absorbing states (detected, stopped, target).
  - `step_probability`: probability of one more move in the current idle period.
  - `build_statespace`: reachable states.
  - `get_transition_function`: transition kernel without protection.
  - `belief_transition`: kernel seen by an attacker with a given belief.
- `GRAPHS`, `Graph_choice`: the three case studies.

**`Stackelberg_regime_assessment.py`**: Stackelberg regime and shared tools.
- `solve_ctr_milp_stackelberg`: optimal deployment (MILP).
- `solve_mdp`, `best_response`: optimal attacker policy by backward induction.
- `evaluate_policy`, `expected_success`: success of a fixed policy, or of a mixture of policies.
- `random_defender`, `shortest_path_placement`: baselines.
- `start_distribution`, `analyze_one_h`, `main`: setup, one budget, full run.

**`Blind_regime_assessment.py`**: Blind regime.
- `blind_belief`, `blind_attacker_policy`: the blind attacker.
- `blind_defender`: best deployment against it.
- `analyze_one_h_blind`, `main`: one budget (including the worst case), full run.

**`Dirichlet_regime_asessment.py`**: Dirichlet regime.
- `make_alpha`, `sample_beliefs`: Dirichlet parameter and belief sampling.
- `belief_policy`, `group_policies`: attacker policy for each belief; identical policies merged.
- `solve_dirichlet_milp`: optimal deployment (MILP).
- `mean_ci`: mean and 95% confidence interval (paired for the gain).
- `analyze_one_h_dirichlet`, `main`: one budget (train/test, gain, worst case), full run.

**`results_io.py`**: `save_results`, `export_graphs`, `write_run_info`, `load_scenario`, `plot_scenarios_side_by_side`.

**`Sensitivity.py`**: `grid` (all belief parameters), `k_convergence` (number of training beliefs).

**`Baselines.py`**: `betweenness_placement`, `min_cut_placement`, `greedy_placement`, `run_graph`.

**`CTR_2023_comp.py`**
- Part 1: reproduction of the 2023 values (`committed_matrix`, `value_committed`).
- Part 2: ablation (`law_2023`, `law_ours`, `value_adaptive_mixed`).
- Part 3: the 2023 game on the present model (`normal_form_value`), checked between the blind and Stackelberg values (`bracketing`).

**`Assumption_1_exhibit.py`**: `exhibit` finds the boundary belief by bisection, builds the sequence converging to it (`N_SEQ = 10`) and checks Theorem 1; the helpers are `L`, `best_against`, `mix` and `empirical_radius`.

**`make_tables.py`**, **`plot_results.py`**: tables and figures from the CSV files.

In case of questions, don't hesitate to get in touch with arnold.kouam_kounchou@jku.at (corresponding author of the article).