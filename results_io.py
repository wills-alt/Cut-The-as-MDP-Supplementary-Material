"""
Save the per-h results of each regime to CSV and draw the three-panel figure per graph.
CSV: csv_results/<scenario>_<graph>.csv, one row per h (column N), scenario in
{"stackelberg", "uninformed", "dirichlet"}.
"""
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import matplotlib.pyplot as plt
from Attack_Graph_Implementation import GRAPHS

RESULTS_DIR = Path("csv_results")
RESULTS_DIR.mkdir(exist_ok=True)

_TITLES = {
    "stackelberg": "Fully informed attacker: Stackelberg",
    "uninformed": "Uninformed attacker: blind",
    "dirichlet": "Partially informed attacker: Dirichlet",
}


def _csv_path(graph_name, scenario):
    return RESULTS_DIR / f"{scenario}_{graph_name}.csv"


def save_results(rows, graph_name, scenario):
    """Write one row per h (all keys of the result dicts) to csv_results/<scenario>_<graph>.csv."""
    path = _csv_path(graph_name, scenario)
    fields = ["graph", "scenario", "N"] + [k for k in rows[0] if k != "h"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"graph": graph_name, "scenario": scenario, "N": r["h"],
                        **{k: v for k, v in r.items() if k != "h"}})
    print(f"  [IO] {scenario} results saved -> {path}")


def export_graphs(folder=RESULTS_DIR / "graphs"):
    """
    Write each case-study graph to graphs/<name>.json (nodes, edges, entries, targets) so the
    instances can be reused without this code.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for key, d in GRAPHS.items():
        path = folder / f"{d['name']}.json"
        with open(path, "w") as f:
            json.dump({"id": key, **d, "edges": [list(e) for e in d["edges"]]}, f, indent=1)
    print(f"  [IO] graphs exported -> {folder}")


def _version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def write_run_info(solver, time_limit, settings, runtimes, path=RESULTS_DIR / "run_info.json"):
    """
    Record the computational environment of a run (solver, versions, hardware, date,
    settings, wall-clock time per regime) in csv_results/run_info.json.
    """
    info = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solver": {"name": {"GRB": "Gurobi", "CBC": "CBC (COIN-OR)"}.get(solver, solver),
                   "gurobipy_version": _version("gurobipy") if solver == "GRB" else None,
                   "time_limit_s": time_limit,
                   "status_required": "OPTIMAL (default relative MIP gap 1e-4)"},
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in ("mip", "numpy", "scipy", "matplotlib", "networkx")},
        "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                     "system": platform.platform()},
        "settings": settings,
        "wall_clock_s": runtimes,
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=1)
    print(f"  [IO] run information saved -> {path}")


def load_scenario(graph_name, scenario):
    """Return {h: {V_optimal, V_random_avg, V_random_std, V_shortest, V_stackelberg}}."""
    path = _csv_path(graph_name, scenario)
    if not path.exists():
        print(f"  [IO] WARNING: {path} not found, skipping.")
        return {}
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v_st = row.get("V_stackelberg", "")
            data[int(row["N"])] = {
                "V_optimal": float(row["V_optimal"]),
                "V_random_avg": float(row["V_random_avg"]),
                "V_random_std": float(row["V_random_std"]),
                "V_shortest": float(row["V_shortest"]),
                "V_stackelberg": float(v_st) if v_st.strip() else None,
            }
    return data


def plot_scenarios_side_by_side(graph_name, scenarios=("stackelberg", "uninformed", "dirichlet"),
                                save_png=True, show=True):
    """One figure per graph, one panel per regime (Fig. 3-5)."""
    loaded = {sc: load_scenario(graph_name, sc) for sc in scenarios}
    loaded = {sc: d for sc, d in loaded.items() if d}
    if not loaded:
        print(f"  [IO] No data for graph {graph_name}.")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5.5), sharey=True)
    for ax, sc in zip(axes, scenarios):
        if sc not in loaded:
            ax.set_visible(False)
            continue
        data = loaded[sc]
        Ns = sorted(data)
        ax.plot(Ns, [data[N]["V_optimal"] for N in Ns], "o-", color="tab:blue",
                linewidth=2.2, markersize=7, label="Optimal defender")
        ax.errorbar(Ns, [data[N]["V_random_avg"] for N in Ns],
                    yerr=[data[N]["V_random_std"] for N in Ns], fmt="^--", color="tab:green",
                    linewidth=2, markersize=7, capsize=4, alpha=0.85,
                    label="Random defender (mean ± std)")
        ax.plot(Ns, [data[N]["V_shortest"] for N in Ns], "D-", color="tab:orange",
                linewidth=2, markersize=6, label="Shortest-path defender")
        if all(data[N]["V_stackelberg"] is not None for N in Ns):
            ax.plot(Ns, [data[N]["V_stackelberg"] for N in Ns], "s-", color="purple",
                    linewidth=2.2, markersize=7, label="Stackelberg deployment")
        ax.set_title(_TITLES[sc], fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(Ns)
        ax.set_xlabel("Number of protections", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    axes[0].set_ylabel("Attacker success probability", fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_png:
        fname = f"Defender_strat_comparison_{GRAPHS[graph_name]['name']}.png"
        plt.savefig(fname, dpi=600, bbox_inches="tight", pad_inches=0.1)
        print(f"  [IO] Figure saved -> {fname}")
    if show:
        plt.show()
    plt.close(fig)