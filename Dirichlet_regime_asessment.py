"""
Dirichlet regime: the attacker holds a private belief p ~ Dir(alpha) about where
the protection is (normalized: sum_i p_i = 1), computes its optimal policy
pi*(p) in the belief kernel P^p, and follows it whatever the true deployment x is.
The defender knows alpha but not p, and minimizes J_alpha(x) = E_p[ L(x, pi*(p)) ].
"""
import time
import numpy as np
from mip import Model, xsum, CONTINUOUS, BINARY, OptimizationStatus
from Attack_Graph_Implementation import Graph_choice
import Stackelberg_regime_assessment as ST
from Stackelberg_regime_assessment import (start_distribution, solve_mdp, best_response,
                                          evaluate_policy, expected_success, random_defender,
                                          shortest_path_placement, solve_ctr_milp_stackelberg,
                                          print_summary, timing_columns, H_VALUES)
from results_io import save_results

LAM, M_CONC = 0.5, 10        # main setting: reconnaissance quality and concentration
K_TRAIN, K_EVAL = 500, 2000  # samples used by the MILP / fresh samples for evaluation
SEED_TRAIN, SEED_EVAL = 1, 2


def make_alpha(graph, h, x_st, lam, M, floor=1e-3):
    """
    Dirichlet parameter alpha = M * m_lam over V_spot:
        m_lam = lam * x_St / h + (1 - lam) / |V_spot|
    lam in [0, 1]: reconnaissance quality (0: uniform mean belief, as the blind attacker;
                   1: mean belief on the Stackelberg deployment x_St);
    M > 0       : concentration (confidence of the attacker in its belief).
    alpha_i >= floor keeps every component strictly positive (needed when lam = 1).
    """
    n = len(graph.V_spot)
    x_st = set(x_st)
    return {i: max(M * (lam * (i in x_st) / h + (1 - lam) / n), floor) for i in graph.V_spot}


def sample_beliefs(graph, alpha, K, seed):
    """
    K i.i.d. beliefs p_k ~ Dir(alpha) over V_spot (fixed seed, reproducible).
    alpha: dict {node: alpha_i} on V_spot. Nodes outside V_spot get belief 0.
    """
    rng = np.random.default_rng(seed)
    a = np.array([alpha[i] for i in graph.V_spot], dtype=float)
    return [dict(zip(graph.V_spot, p)) for p in rng.dirichlet(a, size=K)]


def belief_policy(graph, p, nu):
    """pi*(p): optimal attacker policy in the belief kernel P^p (LP (19), solved exactly)."""
    return solve_mdp(graph, graph.belief_transition(p), nu)


def group_policies(policies):
    """
    Merge identical policies: returns [(policy, weight)], weight = fraction of samples
    inducing that policy. J_alpha(x) = sum_g weight_g * L(x, pi_g) is unchanged.
    """
    groups = {}
    for pi in policies:
        key = tuple(sorted(pi.items()))
        groups[key] = groups.get(key, 0) + 1
    K = len(policies)
    return [(dict(key), n / K) for key, n in groups.items()]


def solve_dirichlet_milp(graph, groups, h, nu):
    """
    Dirichlet MILP (21): min_x sum_g weight_g sum_s nu_s u_{s,g}
        u_{s,g} >= sum_s' P(s'|s, pi_g(s)) w_{s,s',g}       (Bellman evaluation of pi_g)
        w_{s,s',g} = (1 - x_i) u_{s',g},  s = (i, c)        (big-M, M = 1, m = -1; w >= 0)
        u = 0 on SINK and STOP, u = 1 on targets, sum_i x_i <= h, x binary.
    groups: [(policy, weight)] from group_policies.
    """
    S, P = graph.statespace, graph.P
    idx = {s: i for i, s in enumerate(S)}
    zero_states = {graph.SINK, graph.STOP}
    transient = [s for s in S if s[1] is not None]

    m = Model(sense="MIN", solver_name=ST.SOLVER)
    m.verbose = 0
    x = {n: m.add_var(var_type=BINARY) for n in graph.V_spot}
    u = {(g, i): m.add_var(lb=0, ub=1, var_type=CONTINUOUS)
         for g in range(len(groups)) for i in range(len(S))}
    w = {}
    for g, (pi, _) in enumerate(groups):
        for s in transient:
            for t in P[s][pi[s]]:
                w[g, idx[s], idx[t]] = m.add_var(lb=0, ub=1, var_type=CONTINUOUS)

    m.objective = xsum(wt * nu[i] * u[g, i] for g, (_, wt) in enumerate(groups)
                       for i in range(len(S)) if nu[i] > 0)
    m += xsum(x.values()) <= h
    for g, (pi, _) in enumerate(groups):
        for s in zero_states:
            m += u[g, idx[s]] == 0
        for s in graph.TARGETS:
            m += u[g, idx[s]] == 1
        for s in transient:
            i = idx[s]
            m += u[g, i] >= xsum(p * w[g, i, idx[t]] for t, p in P[s][pi[s]].items())

    for (g, i, j), wv in w.items():
        xn = x.get(S[i][0])
        if xn is None:
            m += wv == u[g, j]
        else:
            m += wv <= 1 - xn
            m += wv <= u[g, j] + xn
            m += wv >= u[g, j] - xn

    t0 = time.perf_counter()
    status = m.optimize(max_seconds=ST.TIME_LIMIT)
    runtime = time.perf_counter() - t0
    if status != OptimizationStatus.OPTIMAL:
        raise RuntimeError(f"Dirichlet MILP not solved to optimality: {status}")
    return {"placement": [n for n, v in x.items() if v.x > 0.5],
            "objective": m.objective_value, "milp_runtime_s": runtime}


def sample_groups(graph, alpha, K, seed, nu):
    """Sample K beliefs, compute each attacker's policy pi*(p_k), merge identical ones."""
    return group_policies([belief_policy(graph, p, nu)["policy"]
                           for p in sample_beliefs(graph, alpha, K, seed)])


def mean_ci(graph, groups, x, nu, K, x_ref=None, z=1.96):
    """
    Sample mean and 95% half-width of L(x, pi_k) over the K evaluation beliefs (groups hold
    the frequency of each distinct policy). With x_ref, the PAIRED difference
    L(x_ref, pi_k) - L(x, pi_k) is used instead.
    """
    w = np.array([wt for _, wt in groups])
    d = np.array([evaluate_policy(graph, pi, x, nu) for pi, _ in groups])
    if x_ref is not None:
        d = np.array([evaluate_policy(graph, pi, x_ref, nu) for pi, _ in groups]) - d
    mean = float(w @ d)
    sd = float(np.sqrt(w @ (d - mean) ** 2 * K / (K - 1)))
    return mean, z * sd / np.sqrt(K)


def analyze_one_h_dirichlet(graph, h, nu, lam=LAM, M=M_CONC, K=K_TRAIN, K_eval=K_EVAL,
                            seed_train=SEED_TRAIN, seed_eval=SEED_EVAL):
    """
    All Dirichlet-regime quantities for one budget h.
    x_Dir is optimized on K training samples; every deployment is then evaluated by
    J_alpha estimated on K_eval FRESH samples.
    """
    st = solve_ctr_milp_stackelberg(graph, h, nu)
    alpha = make_alpha(graph, h, st["placement"], lam, M)
    t0 = time.perf_counter()
    train = sample_groups(graph, alpha, K, seed_train, nu)
    attacker_runtime = (time.perf_counter() - t0) / K          # one belief: sample + DP
    test = sample_groups(graph, alpha, K_eval, seed_eval, nu)

    opt = solve_dirichlet_milp(graph, train, h, nu)
    sp_placement, _ = shortest_path_placement(graph, h, nu)
    rnd = random_defender(graph, h, nu, policy=test)
    j_dir, j_dir_ci = mean_ci(graph, test, opt["placement"], nu, K_eval)
    gain, gain_ci = mean_ci(graph, test, opt["placement"], nu, K_eval, x_ref=st["placement"])
    return {
        "h": h, "lam": lam, "M": M, "K": K, "K_eval": K_eval,
        "seed_train": seed_train, "seed_eval": seed_eval,
        "V_optimal": j_dir,                                                 # J(x_Dir), out-of-sample
        "V_optimal_ci": j_dir_ci,                                           # 95% half-width
        "V_in_sample": opt["objective"],                                    # J(x_Dir), training samples
        "gain": gain, "gain_ci": gain_ci,                                   # J(x_St) - J(x_Dir), paired
        "placement_optimal": opt["placement"],
        "milp_runtime_s": opt["milp_runtime_s"],
        "attacker_runtime_s": attacker_runtime,
        "n_policies_train": len(train), "n_policies_eval": len(test),
        "V_stackelberg": expected_success(graph, test, st["placement"], nu),  # J(x_St)
        "placement_stackelberg": st["placement"],
        "V_random_avg": rnd["mean"], "V_random_std": rnd["std"], "random_method": rnd["method"],
        "V_shortest": expected_success(graph, test, sp_placement, nu),
        "placement_shortest": sp_placement,
        "V_worst_case": best_response(graph, opt["placement"], nu)["objective"],  # max_pi L(x_Dir, pi)
        "V_St_value": st["objective"],                                            # V^St
    }


def main(graph_name="1", h_values=H_VALUES, lam=LAM, M=M_CONC, lambda_attack=2,
         lambda_defender=1, defense_pattern="exponential"):
    graph = Graph_choice(graph_name, lambda_attack, lambda_defender, defense_pattern)
    nu = start_distribution(graph)
    rows = [timing_columns(graph) | analyze_one_h_dirichlet(graph, h, nu, lam, M) for h in h_values]
    print_summary(graph, rows, title=f"Dirichlet regime (lambda={lam}, M={M}, out-of-sample J)")
    for r in rows:
        print(f"   h={r['h']}: J(x_Dir)={r['V_optimal']:.4f}±{r['V_optimal_ci']:.4f}  J(x_St)={r['V_stackelberg']:.4f}"
              f"  gain={r['gain']:+.4f}±{r['gain_ci']:.4f}"
              f" | in-sample {r['V_in_sample']:.4f} | worst case {r['V_worst_case']:.4f}"
              f" vs V^St {r['V_St_value']:.4f} | {r['n_policies_train']} policies")
    save_results(rows, graph_name, "dirichlet")
    return rows


# if __name__ == "__main__":
#     main("1")