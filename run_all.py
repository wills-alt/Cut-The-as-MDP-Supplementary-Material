"""
Run the three regimes on the three case studies and save csv_results/*.csv, the graphs
(csv_results/graphs/*.json) and the run environment (csv_results/run_info.json).
Full pipeline:
  python run_all.py        main results of the three regimes 
  python sensitivity.py    experiment B: lambda x M grid and convergence in K 
  python baselines.py      experiment D: stronger baselines 
  python make_tables.py    all LaTeX tables -> csv_results/tables.tex
  python plot_results.py   
Further studies: assumption1_exhibit.py (Assumption 1), ctr2023_comparison.py (CTR 2023).
Solver: set SOLVER in Stackelberg_regime_assessment.py ("GRB" or "CBC").
"""
import time
import Stackelberg_regime_assessment as stackelberg
import Blind_regime_assessment as blind
import Dirichlet_regime_asessment as dirichlet
from Attack_Graph_Implementation import GRAPHS
from results_io import export_graphs, write_run_info

LAMBDA_ATTACK, LAMBDA_DEFENDER, DEFENSE_PATTERN = 2, 1, "exponential"

if __name__ == "__main__":
    timing = dict(lambda_attack=LAMBDA_ATTACK, lambda_defender=LAMBDA_DEFENDER,
                  defense_pattern=DEFENSE_PATTERN)
    export_graphs()
    runtimes = {}
    for name, regime in (("stackelberg", stackelberg), ("blind", blind), ("dirichlet", dirichlet)):
        for g in GRAPHS:
            t0 = time.perf_counter()
            regime.main(g, **timing)
            runtimes[f"{name}_{GRAPHS[g]['name']}"] = round(time.perf_counter() - t0, 2)

    write_run_info(stackelberg.SOLVER, stackelberg.TIME_LIMIT,
                   settings={**timing, "h_values": list(stackelberg.H_VALUES),
                             "nu": "uniform on V \\ F", "V_spot": "V \\ (F u entries)",
                             "dirichlet": {"lam": dirichlet.LAM, "M": dirichlet.M_CONC,
                                           "alpha_floor": 1e-3,
                                           "K_train": dirichlet.K_TRAIN, "K_eval": dirichlet.K_EVAL,
                                           "seed_train": dirichlet.SEED_TRAIN,
                                           "seed_eval": dirichlet.SEED_EVAL},
                             "random_baseline": "exact enumeration if <= 2e6 evaluations, "
                                                "else 1000 samples, seed 42",
                             "ties": "attacker: first action in edge-list order; "
                                     "blind defender: first deployment in lexicographic order; "
                                     "shortest path: node index"},
                   runtimes=runtimes)
    print("\nAll runs complete. Now run: python plot_results.py")