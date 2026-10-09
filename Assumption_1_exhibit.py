"""
Numerical exhibit of Assumption 1 and Theorem 1.

For a case study and a budget h, the script exhibits
  x_St : the Stackelberg deployment,
  x'   : a competing deployment,
  p*   : a boundary belief, where the attacker's belief-optimal policy switches,
  p(n) : a sequence of beliefs converging to p*, all inducing the same policy,
  Delta: the margin  L(x_St, p(n)) - L(x', p(n)) > 0,
and checks Theorem 1 numerically: for alpha_n = M * p(n) with growing M,
J_alpha_n(x') < J_alpha_n(x_St).

Notation: L(x, p) = L(x, pi*(p)) is the realized success of the attacker whose belief is p (policy pi*(p) from LP (19)) against the true deployment x.
"""
import itertools
import numpy as np
from Attack_Graph_Implementation import Graph_choice
import Stackelberg_regime_assessment as ST
from Stackelberg_regime_assessment import start_distribution, evaluate_policy, solve_ctr_milp_stackelberg
from Dirichlet_regime_asessment import (make_alpha, sample_beliefs, belief_policy,
                                                        expected_success, group_policies)
from results_io import RESULTS_DIR
import csv

LAM, M_CONC, K, SEED = 0.5, 10, 500, 1            # beliefs used to find the policy regions
M_LIST = (10, 100, 1000, 10000)                   # concentrations for the Theorem 1 check
K_CHECK, SEED_CHECK = 2000, 7                     # fresh samples for the Theorem 1 check
N_SEQ = 10                                         # length of the sequence p(n) -> p*


def key(policy):
    return tuple(sorted(policy.items()))   # Will be important to test if \pi_1 = \pi_2 for two different policies


def L(graph, x, p, nu):
    """L(x, p): realized success of the attacker with belief p against deployment x."""
    return evaluate_policy(graph, belief_policy(graph, p, nu)["policy"], x, nu)


def best_against(graph, policy, h, nu):
    """x'(pi): deployment minimizing L(x, pi) over all C(|V_spot|, h) deployments."""
    return list(min(itertools.combinations(graph.V_spot, h), key=lambda d: evaluate_policy(graph, policy, d, nu)))


def mix(pa, pb, t):
    """This function will be used to search for a border between two policy regions."""
    return {i: (1 - t) * pa[i] + t * pb[i] for i in pa}


def empirical_radius(graph, p, x_list, nu, rng, r0=0.05, tests=200):
    """
    Largest r (halving from r0) such that L(x, q) = L(x, p) for every x in x_list and
    `tests` random beliefs q in the simplex with ||q - p||_inf <= r (Assumption 1, interior).
    """
    base = [L(graph, x, p, nu) for x in x_list]
    r = r0
    while r > 1e-6:
        ok = True
        for _ in range(tests):
            q = {i: max(p[i] + rng.uniform(-r, r), 1e-12) for i in p}
            s = sum(q.values())
            q = {i: v / s for i, v in q.items()}
            if any(abs(L(graph, x, q, nu) - b) > 1e-12 for x, b in zip(x_list, base)):
                ok = False
                break
        if ok:
            return r
        r /= 2
    return 0.0


def exhibit(graph_name, h):
    graph = Graph_choice(graph_name)
    nu = start_distribution(graph)
    x_st = solve_ctr_milp_stackelberg(graph, h, nu)["placement"]
    alpha = make_alpha(graph, h, x_st, LAM, M_CONC)
    beliefs = sample_beliefs(graph, alpha, K, SEED)

    # 1) policy regions met by the sampled beliefs, and the margin of each policy
    by_policy = {}
    for p in beliefs:
        by_policy.setdefault(key(belief_policy(graph, p, nu)["policy"]), []).append(p)
    print(f"\n{graph.name}, h={h}: x_St={x_st}, {len(by_policy)} distinct attacker policies")
    if len(by_policy) < 2:
        print("  -> a single policy: no boundary belief exists; Assumption 1 fails (negative case).")
        return {"graph": graph.name, "h": h, "x_St": x_st, "n_policies": len(by_policy),
                "Delta": 0.0, "status": "single policy - no boundary"}

    best = None
    for k_, ps in by_policy.items():
        pi = dict(k_)
        xp = best_against(graph, pi, h, nu)
        delta = evaluate_policy(graph, pi, x_st, nu) - evaluate_policy(graph, pi, xp, nu)
        if best is None or delta > best[0]:
            best = (delta, k_, xp, ps)
    delta, key_b, x_prime, ps_b = best
    if delta <= 1e-12:
        print("  -> x_St is optimal against every sampled policy: no positive margin.")
        return {"graph": graph.name, "h": h, "x_St": x_st, "n_policies": len(by_policy),
                "Delta": 0.0, "status": "no positive margin"}

    # 2) p_b in the advantageous region; p_a = nearest sampled belief with another policy
    p_b = ps_b[0]
    others = [p for k_, ps in by_policy.items() if k_ != key_b for p in ps]
    dist = lambda p, q: max(abs(p[i] - q[i]) for i in p)
    p_a = min(others, key=lambda p: dist(p, p_b))

    # 3) bisection on p(t) = (1-t) p_a + t p_b for the switch into policy pi_b
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        if key(belief_policy(graph, mix(p_a, p_b, mid), nu)["policy"]) == key_b:
            hi = mid
        else:
            lo = mid
    t_star = hi
    p_star = mix(p_a, p_b, t_star)  # After t*, π*(p(t))= π(b) and different before
    L_left = [L(graph, x, mix(p_a, p_b, lo), nu) for x in (x_st, x_prime)]

    # 4) sequence p(n) -> p* on the advantageous side, margin and interior radius
    rng = np.random.default_rng(0)
    rows = []
    print(f"  x' = {x_prime},  Delta = {delta:.4f},  boundary at t* = {t_star:.6f} on [p_a, p_b]")
    print(f"  just before p* {p_star}: L(x_St)={L_left[0]:.4f}, L(x')={L_left[1]:.4f}  (other policy)")
    print(f"   n   ||p(n)-p*||_inf   L(x_St,p(n))  L(x',p(n))   Delta_n   r_n")
    for n in range(1, N_SEQ + 1):
        t_n = t_star + (1 - t_star) * 2.0 ** (-n)
        p_n = mix(p_a, p_b, t_n)
        same = key(belief_policy(graph, p_n, nu)["policy"]) == key_b
        l_st, l_xp = L(graph, x_st, p_n, nu), L(graph, x_prime, p_n, nu)
        r_n = empirical_radius(graph, p_n, (x_st, x_prime), nu, rng)
        d_n = dist(p_n, p_star)
        rows.append((n, d_n, l_st, l_xp, l_st - l_xp, r_n, same))
        print(f"  {n:2d}   {d_n:.2e}        {l_st:.4f}       {l_xp:.4f}     {l_st - l_xp:.4f}   {r_n:.1e}"
              + ("" if same else "   (policy differs!)"))

    # 5) Theorem 1 check: alpha_n = M * p(n) for the last p(n), growing M, fresh samples
    p_last = mix(p_a, p_b, t_star + (1 - t_star) * 2.0 ** (-N_SEQ))
    print(f"   M       J(x_St)   J(x')    J(x_St)-J(x')   (alpha = M * p({N_SEQ}), {K_CHECK} samples)")
    checks = []
    for M in M_LIST:
        a_n = {i: max(M * p_last[i], 1e-3) for i in p_last}
        groups = group_policies([belief_policy(graph, p, nu)["policy"]
                                 for p in sample_beliefs(graph, a_n, K_CHECK, SEED_CHECK)])
        j_st, j_xp = expected_success(graph, groups, x_st, nu), expected_success(graph, groups, x_prime, nu)
        checks.append((M, j_st, j_xp))
        print(f"  {M:6d}   {j_st:.4f}   {j_xp:.4f}   {j_st - j_xp:+.4f}")

    # print("\n Summary for advantageous scenario")
    # print(f"Graph: {graph.name}, budget h = {h}")
    # print(f"Stackelberg placement x_St: {x_st}")
    # print(f"Competing placement x':       {x_prime}")
    # print(f"Attacker policy pi_b (on advantageous side):")
    # for s, a in sorted(dict(key_b).items()):
    #     print(f"  {s} -> {a}")
    # print(f"Boundary belief p* (approx):")
    # for i, v in sorted(p_star.items()):
    #     print(f"  {i}: {v:.6f}")
    # print(f"Margin Delta = L(x_St, pi_b) - L(x', pi_b) = {delta:.6f} (> 0)")
    # print(f"Sequence p(n) -> p* has {len(rows)} points, all with same policy pi_b: {all(r[-1] for r in rows)}")
    # M_last, j_st_last, j_xp_last = checks[-1]
    # print(f"Theorem 1 check (M = {M_last}):")
    # print(f"  J_alpha(x_St) = {j_st_last:.6f}")
    # print(f"  J_alpha(x')   = {j_xp_last:.6f}")
    # print(f"  J_alpha(x') - J_alpha(x_St) = {j_xp_last - j_st_last:.6f} (< 0)")
    # print("=> Advantageous scenario explicitly exhibited.")
    # print("================================================\n")

    return {"graph": graph.name, "h": h, "x_St": x_st, "n_policies": len(by_policy),
            "x_prime": x_prime, "Delta": delta, "t_star": t_star,
            "p_star": {i: round(v, 4) for i, v in p_star.items()},
            "sequence": rows, "theorem_check": checks, "status": "exhibited"}


def main(cases=(("1", 2), ("2", 2), ("3", 1), ("3", 3))):
    results = [exhibit(g, h) for g, h in cases]
    path = RESULTS_DIR / "assumption1_exhibit.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "h", "x_St", "n_policies", "x_prime", "Delta", "t_star", "p_star",
                    "J_xSt_M10000", "J_xprime_M10000", "status"])
        for r in results:
            last = r.get("theorem_check", [(None, None, None)])[-1]
            w.writerow([r["graph"], r["h"], r["x_St"], r["n_policies"], r.get("x_prime"),
                        r["Delta"], r.get("t_star"), r.get("p_star"), last[1], last[2], r["status"]])
    print(f"\n  [IO] summary saved -> {path}")
    return results


if __name__ == "__main__":
    main()