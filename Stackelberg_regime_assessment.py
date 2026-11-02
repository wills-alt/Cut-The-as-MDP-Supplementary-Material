"""
Cut-The-Rope MILP Solver for Stackelberg game approach: attacker having full information.
"""

import matplotlib.pyplot as plt
import numpy as np
from Attack_Graph_Implementation import Graph_choice
from mip import Model, xsum, CONTINUOUS, BINARY, OptimizationStatus
import networkx as nx
import itertools, math, random, time  
from results_io import save_results

SOLVER = "CBC"          # "CBC" (open source) or "GRB" (Gurobi);
TIME_LIMIT = 3600        # seconds per MILP
H_VALUES = range(1, 6)  # budgets h shown in the figures

# =============================================================================
# BELIEF CREATION
# =============================================================================

verbose = False

def start_distribution(graph, nodes = None):
    """
    Initial distribution nu over states (n, 0), aligned with graph.statespace.
    Default: uniform over graph.start_nodes = V - F. 
    """
    nodes = graph.start_nodes if nodes is None else nodes
    idx = {s: i for i, s in enumerate(graph.statespace)}
    nu = np.zeros(len(graph.statespace))
    for n in nodes:
        nu[idx[(n, 0)]] = 1.0 / len(nodes)
    return nu


# =============================================================================================
# Stackelberg regime (Section III): the attacker observes the deployment x and best-responds.
# =============================================================================================

def solve_ctr_milp_stackelberg(graph, h, nu):
    """ Stackelberg MILP :  min_x min_v  sum_s nu_s v_s
        v_s >= sum_s' P(s'|s,a) w_{s,s'}          for all transient s, all a   (Bellman)
        w_{s,s'} = (1 - x_i) v_s',  s = (i, c)    (big-M linearization method, M = 1, m = -1)
        v = 0 on SINK and STOP,  v = 1 on targets,  sum_i x_i <= h,  x binary.
    Protection acts on the node being left: from a protected node every move is detected.
    The function returns the protected nodes, the optimal value and the state values.
    """
    
    S, P = graph.statespace, graph.P
    idx = {s: i for i, s in enumerate(S)}
    zero_states = {graph.SINK, graph.STOP}
    absorbing = zero_states | set(graph.TARGETS)

    m = Model(sense="MIN", solver_name = SOLVER)
    m.verbose = 0
    v = [m.add_var(lb=0, ub=1, var_type=CONTINUOUS) for _ in S]
    x = {n: m.add_var(var_type=BINARY) for n in graph.V_spot}

    # w_{s,s'} for every transition s -> s' with positive probability
    w = {}
    for s in S:
        if s in absorbing:
            continue
        for row in P[s].values():
            for t, p in row.items():
                if p > 0 and (idx[s], idx[t]) not in w:
                    w[idx[s], idx[t]] = m.add_var(lb=0, ub=1, var_type=CONTINUOUS)

    m.objective = xsum(nu[i] * v[i] for i in range(len(S)) if nu[i] > 0)
    m += xsum(x.values()) <= h
    for s in zero_states:
        m += v[idx[s]] == 0
    for s in graph.TARGETS:
        m += v[idx[s]] == 1

    # Bellman inequalities, one per action
    for s in S:
        if s in absorbing:
            continue
        i = idx[s]
        for row in P[s].values():
            m += v[i] >= xsum(p * w[i, idx[t]] for t, p in row.items())

    # Linearization of w = (1 - x_node) v_t   (w >= 0 is the variable's lower bound)
    for (i, j), wij in w.items():
        xn = x.get(S[i][0])
        if xn is None:                       # node not protectable: w = v_t
            m += wij == v[j]
        else:
            m += wij <= 1 - xn
            m += wij <= v[j] + xn
            m += wij >= v[j] - xn

    status = m.optimize(max_seconds=TIME_LIMIT)
    if status != OptimizationStatus.OPTIMAL:
        raise RuntimeError(f"Stackelberg MILP not solved to optimality: {status}")

    return {
        "placement": [n for n, var in x.items() if var.x > 0.5],
        "objective": m.objective_value,
        "values": {s: v[idx[s]].x for s in S},
    }


def solve_mdp(graph, kernel, nu, protected=()):
    """ Optimal attacker policy and values for a given kernel, by backward induction
    (exact solution of the LPs (4)-(8), (18), (19)): every transition increases c or is absorbed, so states are processed in decreasing c.
    kernel    : graph.P (true dynamics) or graph.belief_transition(q) (belief dynamics)
    protected : nodes whose departure is detected for sure (deployment x); 
    Ties between actions: first action in edge-list order.
    Returns sum_s nu_s V(s), the values V and the policy.
    """
    protected = set(protected)
    V = {graph.SINK: 0.0, graph.STOP: 0.0}
    V.update({t: 1.0 for t in graph.TARGETS})
    policy = {}
    transient = sorted((s for s in graph.statespace if s[1] is not None),
                       key=lambda s: -s[1])
    for s in transient:
        best_a, best_q = None, -1.0
        for a, row in kernel[s].items():
            q = 0.0 if s[0] in protected else sum(p * V[t] for t, p in row.items())
            if q > best_q + 1e-12:
                best_a, best_q = a, q
        V[s], policy[s] = best_q, best_a
    value = float(sum(nu[i] * V[s] for i, s in enumerate(graph.statespace)))
    return {"objective": value, "values": V, "policy": policy}


def best_response(graph, protected, nu):
    """Attacker who observes the deployment: optimal policy in P^x."""
    return solve_mdp(graph, graph.P, nu, protected)


def evaluate_policy(graph, policy, protected, nu):
    """ Realized success L(x, pi) of a FIXED attacker policy against the true deployment x
    (Bellman evaluation in P^x, by backward induction). Used for attackers who do not
    observe x (blind, Dirichlet), while leaving a protected node leads to SINK.
    """
    protected = set(protected)
    V = {graph.SINK: 0.0, graph.STOP: 0.0}
    V.update({t: 1.0 for t in graph.TARGETS})
    transient = sorted((s for s in graph.statespace if s[1] is not None),
                       key=lambda s: -s[1])
    for s in transient:
        V[s] = 0.0 if s[0] in protected else \
            sum(p * V[t] for t, p in graph.P[s][policy[s]].items())
    return float(sum(nu[i] * V[s] for i, s in enumerate(graph.statespace)))

def expected_success(graph, groups, protected, nu):
    """J(x) = sum_g weight_g * L(x, pi_g) for a mixture of fixed policies [(pi_g, weight_g)]."""
    return sum(wt * evaluate_policy(graph, pi, protected, nu) for pi, wt in groups)

########    EVALUATION SUCCESS PROBABIILITY UNDER RANDOM DEFENDER
### the random baseline is evaluated exactly, by enumerating all $\binom{|V_{\text{spot}}|}{h}$ deployments.

def random_defender(graph, h, nu, policy=None, max_evals=2_000_000, n_samples=1000, seed=42):
    """
    Random-defense baseline: the defender protects h nodes of V_spot drawn uniformly at random.
    policy=None          : the attacker best-responds to the realized deployment (Stackelberg);
    policy=pi            : the attacker follows the fixed policy pi (blind);
    policy=[(pi_g, w_g)] : mixture of fixed policies (Dirichlet), value = expected success.
    All C(|V_spot|, h) deployments are enumerated (exact mean and std) when the number of
    """
    if policy is None:
        value, cost = (lambda d: best_response(graph, d, nu)["objective"]), 1
    elif isinstance(policy, dict):
        value, cost = (lambda d: evaluate_policy(graph, policy, d, nu)), 1
    else:
        value, cost = (lambda d: expected_success(graph, policy, d, nu)), len(policy)

    n = len(graph.V_spot)
    if math.comb(n, h) * cost <= max_evals:
        deployments = itertools.combinations(graph.V_spot, h)
        method = "exact"
    else:
        rng = random.Random(seed)
        deployments = (rng.sample(graph.V_spot, h) for _ in range(n_samples))
        method = f"sampled (n={n_samples}, seed={seed})"
    vals = np.array([value(d) for d in deployments])
    return {"mean": vals.mean(), "std": vals.std(), "n": len(vals), "method": method}




def shortest_path_placement(graph, h, nu):
    """Shortest-path heuristic (Section VI-2). An attacker starting at e follows a shortest path (in edges) to its nearest target, all such paths being equally likely; 
    Score of node j: sigma(j) = sum_e nu_e * (#shortest e->F paths through j) / (#shortest e->F paths).
    Returns the h nodes of V_spot with the highest score (ties by node index) and sigma.
    """
    idx = {s: i for i, s in enumerate(graph.statespace)}
    targets = set(graph.target_nodes)
    sigma = {j: 0.0 for j in graph.V_spot}

    for e in graph.start_nodes:
        w_e = nu[idx[(e, 0)]]
        # BFS distances from e (targets absorbing)
        d, order = {e: 0}, [e]
        for u in order:
            if u in targets:
                continue
            for v in graph.adjacency[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    order.append(v)
        reach = [t for t in targets if t in d]
        if not reach:
            continue
        L = min(d[t] for t in reach)
        nearest = {t for t in reach if d[t] == L}
        # f[u]: number of shortest e->u paths;  b[u]: number of shortest u->nearest paths
        f = {e: 1}
        for u in order:
            if u in targets:
                continue
            for v in graph.adjacency[u]:
                if d.get(v) == d[u] + 1:
                    f[v] = f.get(v, 0) + f.get(u, 0)
        b = {t: 1 for t in nearest}
        for u in reversed(order):
            if u not in targets:
                b[u] = sum(b.get(v, 0) for v in graph.adjacency[u] if d.get(v) == d[u] + 1)
        total = sum(f.get(t, 0) for t in nearest)
        for j in sigma:
            sigma[j] += w_e * f.get(j, 0) * b.get(j, 0) / total

    ranked = sorted(sigma, key=lambda j: (-sigma[j], j))
    return ranked[:h], sigma


#############   Main analysis function  ##########################################

def analyze_one_h(graph, h, nu):
    """ All Stackelberg-regime quantities for one budget h: optimal deployment, random baseline and shortest-path baseline, each evaluated against
    the best-responding attacker under nu.
    """
    t0 = time.perf_counter()
    opt = solve_ctr_milp_stackelberg(graph, h, nu)
    runtime = time.perf_counter() - t0

    # consistency check: re-evaluating the MILP deployment must give the MILP value
    check = best_response(graph, opt["placement"], nu)["objective"]
    if abs(check - opt["objective"]) > 1e-6:
        raise RuntimeError(f"MILP value {opt['objective']} != best response {check} check the error in the MILP resolution")

    rnd = random_defender(graph, h, nu)
    sp_placement, _ = shortest_path_placement(graph, h, nu)
    return {
        "h": h,
        "V_optimal": opt["objective"],
        "placement_optimal": opt["placement"],
        "milp_runtime_s": runtime,
        "V_random_avg": rnd["mean"],
        "V_random_std": rnd["std"],
        "random_method": rnd["method"],
        "V_shortest": best_response(graph, sp_placement, nu)["objective"],
        "placement_shortest": sp_placement,
    }


def timing_columns(graph):
    """Timing parameters written in every CSV row (reproducibility, R1)."""
    return {"lambda_attack": graph.lambda_attack, "lambda_defender": graph.lambda_defender,
            "defense_pattern": graph.defense_pattern}
 
 
def print_summary(graph, rows, title="Stackelberg regime (attacker best-responds)"):
    print(f"\n{graph.name} - {title}")
    print(f"{'h':>2} {'optimal':>9} {'random':>17} {'shortest':>9}   optimal / shortest-path deployment")
    for r in rows:
        print(f"{r['h']:>2} {r['V_optimal']:>9.4f} {r['V_random_avg']:>9.4f} ±{r['V_random_std']:.4f}"
              f" {r['V_shortest']:>9.4f}   {r['placement_optimal']} / {r['placement_shortest']}")
 
 
def main(graph_name="1", h_values=H_VALUES, lambda_attack=2, lambda_defender=1,
         defense_pattern="exponential"):
    graph = Graph_choice(graph_name, lambda_attack, lambda_defender, defense_pattern)
    nu = start_distribution(graph)
    rows = [timing_columns(graph) | analyze_one_h(graph, h, nu) for h in h_values]
    print_summary(graph, rows)
    save_results(rows, graph_name, "stackelberg")
    return rows



# if __name__ == "__main__":
#     main("1")

