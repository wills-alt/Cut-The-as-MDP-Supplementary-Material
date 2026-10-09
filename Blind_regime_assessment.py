# Uninformed_Optimization.py
"""
Blind regime: the attacker does not observe the deployment. It believes that
each protectable node is protected independently with probability h / |V_spot|, computes
its optimal policy pi^N in the belief kernel P^Blind, and follows it whatever x is.
"""

import itertools
import time
from Attack_Graph_Implementation import Graph_choice
from Stackelberg_regime_assessment import (start_distribution, solve_mdp, best_response,
                                          evaluate_policy, random_defender,
                                          shortest_path_placement, print_summary, timing_columns,
                                          H_VALUES)
from results_io import save_results


def blind_belief(graph, h):
    """Blind belief: q(i) = h / |V_spot| on V_spot, 0 elsewhere."""
    return {i: h / len(graph.V_spot) for i in graph.V_spot}

# =============================================================================
# EVALUATE ATTACKER POLICY UNDER SPECIFIC DEFENDER DEPLOYMENT
# =============================================================================

def blind_attacker_policy(graph, h, nu):
    """pi^N: optimal policy in the belief kernel P^Blind (LP (18))."""
    return solve_mdp(graph, graph.belief_transition(blind_belief(graph, h)), nu)



def blind_defender(graph, policy, h, nu):
    """Defender's best deployment against the fixed blind policy:
    all C(|V_spot|, h) deployments are enumerated; ties -> first in lexicographic order.
    """
    best_val, best_x = float("inf"), None
    for d in itertools.combinations(graph.V_spot, h):
        val = evaluate_policy(graph, policy, d, nu)
        if val < best_val - 1e-12:
            best_val, best_x = val, list(d)
    return {"placement": best_x, "objective": best_val}


def analyze_one_h_blind(graph, h, nu):
    """
    All blind-regime quantities for one budget h.
    Every deployment is evaluated against the SAME fixed policy pi^N.
    V_worst_case is the success of x^N if the attacker could observe it and best-respond.
    """
    t0 = time.perf_counter()
    attacker = blind_attacker_policy(graph, h, nu)
    attacker_runtime = time.perf_counter() - t0
    pi = attacker["policy"]
    t0 = time.perf_counter()
    opt = blind_defender(graph, pi, h, nu)
    runtime = time.perf_counter() - t0
    rnd = random_defender(graph, h, nu, policy=pi)
    sp_placement, _ = shortest_path_placement(graph, h, nu)
    return {
        "h": h,
        "V_optimal": opt["objective"],
        "placement_optimal": opt["placement"],
        "defender_runtime_s": runtime,
        "attacker_runtime_s": attacker_runtime,
        "V_random_avg": rnd["mean"],
        "V_random_std": rnd["std"],
        "random_method": rnd["method"],
        "V_shortest": evaluate_policy(graph, pi, sp_placement, nu),
        "placement_shortest": sp_placement,
        "V_attacker_belief": attacker["objective"],
        "V_worst_case": best_response(graph, opt["placement"], nu)["objective"],
    }



def main(graph_name="1", h_values=H_VALUES, lambda_attack=2, lambda_defender=1,
         defense_pattern="exponential"):
    graph = Graph_choice(graph_name, lambda_attack, lambda_defender, defense_pattern)
    nu = start_distribution(graph)
    rows = [timing_columns(graph) | analyze_one_h_blind(graph, h, nu) for h in h_values]
    print_summary(graph, rows, title="Blind regime (attacker follows pi^N)")
    for r in rows:
        print(f"   h={r['h']}: worst case of x^N (attacker observes it) = {r['V_worst_case']:.4f}")
    save_results(rows, graph_name, "uninformed")
    return rows

# if __name__ == "__main__":
#     main()