"""
Comparison with the Cut-The-Rope model of Rass et al. (2023), Computers & Security. The 2023 model is re-implemented from the authors' public code
(github.com/jku-lit-scsl/ComputersAndSecurity_RoboticsCaseStudies_Cut-The-Rope, ctr-core_1.R):
  - all targets are contracted into one, and the attacker picks ONE path for all avatars;
  - avatars sit on every node except the entry and the targets, with equal weights;
    an avatar that is not on the chosen path does not move (scores 0);
  - the number of steps N is geometric, truncated to the route and renormalized
    (randomSteps in experiment_2.R), so an avatar at distance L reaches the target w.p.
    P(N = L) / P(N <= L);
  - the defender checks one node, chosen by a mixed strategy (zero-sum matrix game).
We thus propose the following:
    ablation: the modeling changes, one at a time, from the 2023 model to the present one
        (MARA, h = 1);
    the 2023 game solved on the present model (normal form: the attacker knows the
        defender's mixed strategy, not the deployment drawn), for all graphs and budgets,
        and compared with the blind and Stackelberg regimes: V^Blind <= V^NF <= V^St.
Output: csv_results/ctr2023_comparison.csv (all parts), csv_results/ctr2023_bracketing.csv 
"""

import csv
import itertools
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from Attack_Graph_Implementation import GRAPHS, Graph_choice
from Stackelberg_regime_assessment import (start_distribution, solve_ctr_milp_stackelberg,
                                          H_VALUES)
from Blind_regime_assessment import blind_attacker_policy, blind_defender
from results_io import RESULTS_DIR

LAMBDA = 2.0
REPORTED_2023 = {1: 0.074, 2: 0.057, 3: 0.046}   # experiment_2.log, MARA, lambda_D = 1, 2, 3


# ============================================================================= Parts 1-2
def mara():
    d = GRAPHS["1"]
    adj = {}
    for a, b in d["edges"]:
        adj.setdefault(a, []).append(b)
    entry, targets = d["entry_nodes"][0], set(d["target_nodes"])
    spots = [n for n in d["nodes"] if n not in targets and n != entry]   # 2023: AS1

    def paths_from(u):
        if u in targets:
            return [[u]]
        return [[u] + p for v in adj.get(u, []) for p in paths_from(v)]
    return d, entry, targets, spots, paths_from(entry)


def law_2023(L, r):
    """2023: truncated geometric, renormalized on the route: P(N = L) / P(N <= L)."""
    pdf = [(1 - r) * r ** n for n in range(L + 1)]
    return pdf[L] / sum(pdf)


def law_ours(L, r):
    """Present paper (Lemma 1): P(N >= L) = r^L."""
    return r ** L


def success(sub_route, spot, law, r):
    """Avatar walking sub_route (from its position to the target); blocked if it meets the spot."""
    return 0.0 if spot in sub_route else law(len(sub_route) - 1, r)


def committed_matrix(avatars, routes, spots, law, r):
    """A[s, path]: average success when ALL avatars follow `path` and the defender checks s."""
    A = np.zeros((len(spots), len(routes)))
    for i, s in enumerate(spots):
        for j, pa in enumerate(routes):
            A[i, j] = np.mean([success(pa[pa.index(a):], s, law, r) if a in pa else 0.0
                               for a in avatars])
    return A


def value_committed(A):
    """min_x max_path x^T A  (mixed defender vs one committed path)."""
    m, n = A.shape
    res = linprog(np.r_[np.zeros(m), 1], A_ub=np.c_[A.T, -np.ones(n)], b_ub=np.zeros(n),
                  A_eq=[np.r_[np.ones(m), 0]], b_eq=[1], bounds=[(0, None)] * m + [(None, None)])
    return res.fun, res.x[:m]


def value_adaptive_mixed(avatars, routes, spots, law, r):
    """
    Each avatar chooses its own path, without observing the check (normal-form game):
    min_x sum_a z_a / |avatars|,  z_a >= sum_s x_s success_a(path, s) for every path through a.
    """
    m, n_a = len(spots), len(avatars)
    c = np.r_[np.zeros(m), np.ones(n_a) / n_a]
    A_ub, b_ub = [], []
    for k, a in enumerate(avatars):
        for pa in (p for p in routes if a in p):
            row = np.zeros(m + n_a)
            row[:m] = [success(pa[pa.index(a):], s, law, r) for s in spots]
            row[m + k] = -1
            A_ub.append(row); b_ub.append(0.0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[np.r_[np.ones(m), np.zeros(n_a)]], b_eq=[1],
                  bounds=[(0, None)] * (m + n_a))
    return res.fun, res.x[:m]


# ============================================================================= Part 3
def attack_paths(graph, s):
    """All paths from node s to the first target reached (targets are absorbing)."""
    if s in graph.target_nodes:
        return [[s]]
    return [[s] + p for v in graph.adjacency[s] for p in attack_paths(graph, v)]


def normal_form_value(graph, h, nu):
    """
    Value of the zero-sum normal-form game on the present model (the 2023 game, generalized):
    the defender mixes over all deployments d of h nodes of V_spot; the attacker knows the
    mixture sigma but not the deployment drawn, and from each start node s follows a path p
    (a deterministic policy; it learns nothing during the round except that it is alive).
        min_sigma  sum_s nu_s z_s
        z_s >= sum_d sigma_d * Pr(N >= |p|) * 1[p leaves no node of d]    for all s, p from s
    Returns the value, the equilibrium mixture {deployment: probability} and the LP size.
    """
    idx = {s: i for i, s in enumerate(graph.statespace)}
    D = list(itertools.combinations(graph.V_spot, h))
    starts = graph.start_nodes
    reach = [1.0]                                        # reach[L] = Pr(N >= L)
    for c in range(graph.c_max + 1):
        reach.append(reach[-1] * graph.step_probability(c))

    rows, cols, vals, n_rows = [], [], [], 0
    for k, s in enumerate(starts):
        for p in attack_paths(graph, s):
            left = set(p[:-1])                           # nodes the attacker departs from
            for j, d in enumerate(D):
                if left.isdisjoint(d):
                    rows.append(n_rows); cols.append(j); vals.append(reach[len(p) - 1])
            rows.append(n_rows); cols.append(len(D) + k); vals.append(-1.0)
            n_rows += 1
    A = csr_matrix((vals, (rows, cols)), shape=(n_rows, len(D) + len(starts)))
    weight = np.array([nu[idx[(s, 0)]] for s in starts])
    res = linprog(np.r_[np.zeros(len(D)), weight], A_ub=A, b_ub=np.zeros(n_rows),
                  A_eq=csr_matrix(np.r_[np.ones(len(D)), np.zeros(len(starts))]), b_eq=[1],
                  bounds=[(0, None)] * (len(D) + len(starts)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"Normal-form LP failed: {res.message}")
    sigma = {d: float(res.x[j]) for j, d in enumerate(D) if res.x[j] > 1e-6}
    return res.fun, sigma, n_rows, len(D)


def bracketing(graphs=tuple(GRAPHS), h_values=H_VALUES):
    """Part 3: V^Blind <= V^NF <= V^St on every graph and budget."""
    rows = []
    for g in graphs:
        graph = Graph_choice(g)
        nu = start_distribution(graph)
        for h in h_values:
            v_st = solve_ctr_milp_stackelberg(graph, h, nu)["objective"]
            pi = blind_attacker_policy(graph, h, nu)["policy"]
            v_bl = blind_defender(graph, pi, h, nu)["objective"]
            t0 = time.perf_counter()
            v_nf, sigma, n_paths, n_dep = normal_form_value(graph, h, nu)
            t = time.perf_counter() - t0
            if not (v_bl <= v_nf + 1e-9 and v_nf <= v_st + 1e-9):
                raise RuntimeError(f"Bracketing violated on {graph.name}, h={h}")
            mix = {str(list(d)): round(p, 3) for d, p in sorted(sigma.items(), key=lambda kv: -kv[1])}
            rows.append({"graph": graph.name, "h": h, "V_blind": v_bl, "V_normal_form": v_nf,
                         "V_stackelberg": v_st, "equilibrium_mixture": mix,
                         "n_paths": n_paths, "n_deployments": n_dep, "lp_runtime_s": t})
            print(f"  {graph.name:8s} h={h}  blind {v_bl:.4f} <= NF {v_nf:.4f} <= St {v_st:.4f}"
                  f"   sigma {mix}   ({n_dep} deployments, {t:.1f}s)", flush=True)
    return rows


# ============================================================================= main
def main():
    d, entry, targets, spots, routes = mara()
    av_2023 = [n for n in d["nodes"] if n not in targets and n != entry]   # V \ ({entry} u F)
    av_ours = [n for n in d["nodes"] if n not in targets]                  # V \ F  (our nu)
    fmt = lambda x: {s: round(float(p), 3) for s, p in zip(spots, x) if p > 1e-3}
    out = []

    print("Part 1. Reproduction of the 2023 MARA results (exponential defender, lambda = 2):")
    for lam_d, reported in REPORTED_2023.items():
        r = LAMBDA / (LAMBDA + lam_d)
        v, x = value_committed(committed_matrix(av_2023, routes, spots, law_2023, r))
        print(f"  lambda_D={lam_d}: re-implementation {v:.4f}  vs  reported {reported}   defense {fmt(x)}")
        out.append(["reproduction", f"lambda_D={lam_d}", "", round(v, 4), reported, fmt(x)])

    print("\nPart 2. Ablation, MARA, lambda = 2, lambda_D = 1, one check:")
    r = LAMBDA / (LAMBDA + 1.0)
    steps = [("0  2023 CTR model", value_committed(committed_matrix(av_2023, routes, spots, law_2023, r))),
             ("1  step law P(N >= L) (Lemma 1)",
              value_committed(committed_matrix(av_2023, routes, spots, law_ours, r))),
             ("2  + one path per starting node instead of one shared route",
              value_adaptive_mixed(av_2023, routes, spots, law_ours, r))]
    for name, (v, x) in steps:
        print(f"  {v:.4f}   {name}   defense {fmt(x)}")
        out.append(["ablation", name, "", round(v, 4), "", fmt(x)])
    # checks: adding the entry to the starting nodes (present nu) does not change step 2,
    # and step 2 is the normal-form game of the present model 
    v_entry = value_adaptive_mixed(av_ours, routes, spots, law_ours, r)[0]
    g = Graph_choice("1")
    v_nf = normal_form_value(g, 1, start_distribution(g))[0]
    if abs(v_entry - steps[-1][1][0]) > 1e-6 or abs(v_nf - steps[-1][1][0]) > 1e-6:
        raise RuntimeError("Step 2 differs with the entry included or from normal_form_value(MARA, h=1)")
    print(f"  check: with the entry among the starting nodes (present nu): {v_entry:.4f} (unchanged)")

    print("\nPart 3. The 2023 game on the present model vs the blind and Stackelberg regimes:")
    brackets = bracketing()
    path = RESULTS_DIR / "ctr2023_bracketing.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brackets[0]))
        w.writeheader()
        w.writerows(brackets)
    print(f"\n  [IO] saved -> {path}")
    for row in brackets:
        out.append(["bracketing", row["graph"], row["h"],
                    f"{row['V_blind']:.4f} <= {row['V_normal_form']:.4f} <= {row['V_stackelberg']:.4f}",
                    "", row["equilibrium_mixture"]])

    path = RESULTS_DIR / "ctr2023_comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "setting", "h", "value", "reported_2023", "defense"])
        w.writerows(out)
    print(f"\n  [IO] saved -> {path}")


if __name__ == "__main__":
    main()