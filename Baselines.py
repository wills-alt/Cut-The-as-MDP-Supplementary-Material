"""
Experiment to study stronger baselines, compared with the optimal deployment and the
shortest-path baseline in the three regimes. We then choose the following
  betweenness : top-h nodes of V_spot by directed betweenness centrality;
  min cut     : minimum node cut separating the entries from the targets, completed or truncated by betweenness to exactly h nodes;
  greedy      : adds, one at a time, the node that most decreases the regime's objective.
Each deployment is evaluated as the optimal one: against the
best-responding attacker (Stackelberg), against pi^N (blind), and on K' = 2000 fresh beliefs
(Dirichlet). Output: csv_results/baselines.csv.
"""
import csv
import networkx as nx
from Attack_Graph_Implementation import GRAPHS, Graph_choice
from Stackelberg_regime_assessment import (start_distribution, solve_ctr_milp_stackelberg,
                                          best_response, evaluate_policy, expected_success,
                                          shortest_path_placement, H_VALUES)
from Blind_regime_assessment import blind_attacker_policy, blind_defender
from Dirichlet_regime_asessment import (make_alpha, sample_groups,
                                                        solve_dirichlet_milp, LAM, M_CONC,
                                                        K_TRAIN, K_EVAL, SEED_TRAIN, SEED_EVAL)
from results_io import RESULTS_DIR

METHODS = ("optimal", "shortest_path", "betweenness", "min_cut", "greedy")


def digraph(graph):
    """networkx DiGraph of the attack graph."""
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from((u, v) for u, v in graph.edges if u not in graph.target_nodes)
    return G


def betweenness_ranking(graph):
    """V_spot sorted by decreasing directed betweenness centrality (ties: node index)."""
    bc = nx.betweenness_centrality(digraph(graph), normalized=False)
    return sorted(graph.V_spot, key=lambda j: (-bc[j], j))


def betweenness_placement(graph, h):
    return betweenness_ranking(graph)[:h]


def min_cut_placement(graph, h):
    """
    Minimum node cut between the entries and the targets: every node v is split into
    (v,'in') -> (v,'out') with capacity 1 if v is in V_spot (protectable), infinite otherwise.
    Edges from an entry directly into a target cannot be blocked by any deployment and are
    left out. If the cut has c <= h nodes, the h - c remaining protections go to the next nodes by
    betweenness; if c > h, the h cut nodes with the highest betweenness are kept.
    Returns the deployment and the size c of the minimum cut.
    """
    spot = set(graph.V_spot)
    F = nx.DiGraph()
    for v in graph.nodes:
        F.add_edge((v, "in"), (v, "out"), capacity=1 if v in spot else float("inf"))
    for u, v in digraph(graph).edges:
        if u in graph.entry_nodes and v in graph.target_nodes:
            continue
        F.add_edge((u, "out"), (v, "in"), capacity=float("inf"))
    for e in graph.entry_nodes:
        F.add_edge("s", (e, "in"), capacity=float("inf"))
    for t in graph.target_nodes:
        F.add_edge((t, "out"), "t", capacity=float("inf"))
    _, (reach, _) = nx.minimum_cut(F, "s", "t")
    cut = {v for v in spot if (v, "in") in reach and (v, "out") not in reach}
    rank = betweenness_ranking(graph)
    if len(cut) >= h:
        return [j for j in rank if j in cut][:h], len(cut)
    return sorted(cut, key=rank.index) + [j for j in rank if j not in cut][:h - len(cut)], len(cut)


def greedy_placement(graph, h, objective):
    """Add, h times, the node of V_spot minimizing objective(deployment) (ties: node index)."""
    x = []
    for _ in range(h):
        x.append(min((j for j in graph.V_spot if j not in x), key=lambda j: (objective(x + [j]), j)))
    return x


def run_graph(g, h_values=H_VALUES):
    graph = Graph_choice(g)
    nu = start_distribution(graph)
    rows = []

    def record(regime, h, placements, value):
        for method, x in placements.items():
            rows.append({"graph": graph.name, "regime": regime, "h": h, "method": method,
                         "value": value(x), "placement": sorted(x)})

    for h in h_values:
        common = {"shortest_path": shortest_path_placement(graph, h, nu)[0],
                  "betweenness": betweenness_placement(graph, h),
                  "min_cut": min_cut_placement(graph, h)[0]}

        # Stackelberg: the attacker observes x and best-responds
        st_value = lambda x: best_response(graph, x, nu)["objective"]
        x_st = solve_ctr_milp_stackelberg(graph, h, nu)["placement"]
        record("stackelberg", h, {"optimal": x_st, **common,
                                  "greedy": greedy_placement(graph, h, st_value)}, st_value)

        # Blind: every deployment against the same fixed policy pi^N
        pi = blind_attacker_policy(graph, h, nu)["policy"]
        bl_value = lambda x: evaluate_policy(graph, pi, x, nu)
        record("blind", h, {"optimal": blind_defender(graph, pi, h, nu)["placement"], **common,
                            "greedy": greedy_placement(graph, h, bl_value)}, bl_value)

        # Dirichlet: built on the training beliefs, evaluated on the fresh test beliefs
        alpha = make_alpha(graph, h, x_st, LAM, M_CONC)
        train = sample_groups(graph, alpha, K_TRAIN, SEED_TRAIN, nu, h)
        test = sample_groups(graph, alpha, K_EVAL, SEED_EVAL, nu, h)
        x_dir = solve_dirichlet_milp(graph, train, h, nu)["placement"]
        greedy_dir = greedy_placement(graph, h, lambda x: expected_success(graph, train, x, nu))
        record("dirichlet", h, {"optimal": x_dir, **common, "greedy": greedy_dir},
               lambda x: expected_success(graph, test, x, nu))

    for regime in ("stackelberg", "blind", "dirichlet"):
        print(f"\n{graph.name} - {regime}")
        print(f"{'h':>2} " + " ".join(f"{m:>13}" for m in METHODS))
        for h in h_values:
            v = {r["method"]: r["value"] for r in rows if r["regime"] == regime and r["h"] == h}
            print(f"{h:>2} " + " ".join(f"{v[m]:>13.4f}" for m in METHODS))
    return rows


def main(graphs=tuple(GRAPHS)):
    for g in graphs:
        graph = Graph_choice(g)
        print(f"{graph.name}: minimum node cut entries -> targets has "
              f"{min_cut_placement(graph, 1)[1]} node(s)")
    rows = [r for g in graphs for r in run_graph(g)]
    path = RESULTS_DIR / "baselines.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  [IO] saved -> {path}")
    return rows


if __name__ == "__main__":
    main()