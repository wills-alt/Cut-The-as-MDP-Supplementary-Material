"""
Experiment: sensitivity of the Dirichlet regime.
  1-  grid lambda in {0, 0.5, 1} x M in {1, 10, 100}, all graphs, h = 1..5
       (train K = 500, seed 1; evaluation on K' = 2000 fresh beliefs, seed 2);
  2-  convergence in K at the main setting (lambda = 0.5, M = 10): K in K_LIST, N_SEEDS
      training seeds per K, all evaluated on the SAME test set (K' = 2000, seed 2).
Outputs: csv_results/sensitivity_grid.csv, csv_results/sensitivity_K.csv.
"""
import csv
import time
import numpy as np
from Attack_Graph_Implementation import GRAPHS, Graph_choice
from Stackelberg_regime_assessment import (start_distribution, solve_ctr_milp_stackelberg,
                                          best_response, H_VALUES)
from Dirichlet_regime_asessment import (make_alpha, sample_groups,
                                                        solve_dirichlet_milp, mean_ci,
                                                        LAM, M_CONC, K_TRAIN, K_EVAL,
                                                        SEED_TRAIN, SEED_EVAL)
from results_io import RESULTS_DIR

LAM_LIST, M_LIST = (0.0, 0.5, 1.0), (1, 10, 100)
K_LIST, N_SEEDS = (50, 100, 250, 500, 1000), 5
K_GRAPHS = ("1", "3")


def write_csv(rows, name):
    path = RESULTS_DIR / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  [IO] saved -> {path}")


def grid(graphs=tuple(GRAPHS), h_values=H_VALUES):
    """B1: one row per (graph, lambda, M, h)."""
    rows = []
    for g in graphs:
        graph = Graph_choice(g)
        nu = start_distribution(graph)
        for h in h_values:
            st = solve_ctr_milp_stackelberg(graph, h, nu)
            for lam in LAM_LIST:
                for M in M_LIST:
                    alpha = make_alpha(graph, h, st["placement"], lam, M)
                    train = sample_groups(graph, alpha, K_TRAIN, SEED_TRAIN, nu, h)
                    test = sample_groups(graph, alpha, K_EVAL, SEED_EVAL, nu, h)
                    opt = solve_dirichlet_milp(graph, train, h, nu)
                    j, j_ci = mean_ci(graph, test, opt["placement"], nu, K_EVAL)
                    j_st, _ = mean_ci(graph, test, st["placement"], nu, K_EVAL)
                    gain, gain_ci = mean_ci(graph, test, opt["placement"], nu, K_EVAL,
                                            x_ref=st["placement"])
                    rows.append({
                        "graph": graph.name, "h": h, "lam": lam, "M": M,
                        "J_dir": j, "J_dir_ci": j_ci, "J_st": j_st, "gain": gain, "gain_ci": gain_ci,
                        "worst_case": best_response(graph, opt["placement"], nu)["objective"],
                        "V_St": st["objective"],
                        "placement_dir": opt["placement"], "placement_st": st["placement"],
                        "n_policies_train": len(train), "milp_runtime_s": opt["milp_runtime_s"]})
                    r = rows[-1]
                    print(f"{graph.name:8s} h={h} lam={lam:.1f} M={M:<4d} J_dir={j:.4f}±{j_ci:.4f} "
                          f"J_st={j_st:.4f} gain={gain:+.4f}±{gain_ci:.4f} worst={r['worst_case']:.4f} "
                          f"V_St={st['objective']:.4f} pol={len(train)} t={opt['milp_runtime_s']:.1f}s",
                          flush=True)
    write_csv(rows, "sensitivity_grid.csv")
    return rows


def k_convergence(graphs=K_GRAPHS, h_values=H_VALUES, lam=LAM, M=M_CONC):
    """B2: one row per (graph, h, K, seed); all deployments evaluated on one test set."""
    rows = []
    for g in graphs:
        graph = Graph_choice(g)
        nu = start_distribution(graph)
        for h in h_values:
            st = solve_ctr_milp_stackelberg(graph, h, nu)
            alpha = make_alpha(graph, h, st["placement"], lam, M)
            test = sample_groups(graph, alpha, K_EVAL, SEED_EVAL, nu, h)
            for K in K_LIST:
                for seed in range(101, 101 + N_SEEDS):          # disjoint from seeds 1, 2
                    t0 = time.perf_counter()
                    train = sample_groups(graph, alpha, K, seed, nu, h)
                    opt = solve_dirichlet_milp(graph, train, h, nu)
                    j, _ = mean_ci(graph, test, opt["placement"], nu, K_EVAL)
                    rows.append({"graph": graph.name, "h": h, "K": K, "seed": seed,
                                 "J_in_sample": opt["objective"], "J_out_of_sample": j,
                                 "placement_dir": sorted(opt["placement"]),
                                 "n_policies_train": len(train),
                                 "runtime_s": time.perf_counter() - t0})
            ref = {tuple(r["placement_dir"]) for r in rows
                   if r["graph"] == graph.name and r["h"] == h and r["K"] == K_LIST[-1]}
            for K in K_LIST:
                sel = [r for r in rows if r["graph"] == graph.name and r["h"] == h and r["K"] == K]
                js = [r["J_out_of_sample"] for r in sel]
                agree = np.mean([tuple(r["placement_dir"]) in ref for r in sel])
                print(f"{graph.name:8s} h={h} K={K:<5d} J_out mean={np.mean(js):.4f} "
                      f"range=[{min(js):.4f}, {max(js):.4f}]  same x_Dir as K={K_LIST[-1]}: "
                      f"{agree:.0%}", flush=True)
    write_csv(rows, "sensitivity_K.csv")
    return rows


if __name__ == "__main__":
    grid()
    #k_convergence()
    k_convergence(h_values=range(1, 4))