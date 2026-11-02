"""
Complete numerical tables generated from csv_results/*.csv. Output: csv_results/tables.tex. Run this code after run_all.py.
"""
import ast
import csv
from Attack_Graph_Implementation import GRAPHS, Graph_choice
from results_io import RESULTS_DIR

D = 4  # decimals


def load(scenario, g):
    with open(RESULTS_DIR / f"{scenario}_{g}.csv", newline="") as f:
        return list(csv.DictReader(f))


def f(x, d=D):
    return f"{float(x):.{d}f}"


def pm(x, e):
    return f"{f(x)} $\\pm$ {f(e)}"


def dep(s):
    return "\\{" + ", ".join(map(str, sorted(ast.literal_eval(s)))) + "\\}"


def table(caption, label, header, rows_by_graph):
    ncol = len(header) + 2
    out = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{ll" + "c" * len(header) + "}", "\\toprule",
           "Graph & $h$ & " + " & ".join(header) + " \\\\", "\\midrule"]
    for k, (name, rows) in enumerate(rows_by_graph):
        for i, cells in enumerate(rows):
            out.append(f"{name if i == 0 else ''} & {i + 1} & " + " & ".join(cells) + " \\\\")
        if k < len(rows_by_graph) - 1:
            out.append(f"\\cmidrule(lr){{1-{ncol}}}")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def stackelberg_table():
    data = [(GRAPHS[g]["name"],
             [[f(r["V_optimal"]), dep(r["placement_optimal"]), pm(r["V_random_avg"], r["V_random_std"]),
               f(r["V_shortest"]), dep(r["placement_shortest"])] for r in load("stackelberg", g)])
            for g in GRAPHS]
    return table("Stackelberg regime: attacker success under the optimal deployment $x^{\\mathrm{St}}$, "
                 "the random deployment (mean $\\pm$ std over all $\\binom{|V_{\\mathrm{spot}}|}{h}$ "
                 "deployments) and the shortest-path deployment $x^{\\mathrm{SP}}$.",
                 "tab:stackelberg",
                 ["$V^{\\mathrm{St}}$", "$x^{\\mathrm{St}}$", "Random", "Shortest path",
                  "$x^{\\mathrm{SP}}$"], data)


def blind_table():
    data = [(GRAPHS[g]["name"],
             [[f(r["V_optimal"]), dep(r["placement_optimal"]), pm(r["V_random_avg"], r["V_random_std"]),
               f(r["V_shortest"]), f(r["V_worst_case"])] for r in load("uninformed", g)])
            for g in GRAPHS]
    return table("Blind regime: success of the blind policy $\\pi^N$ against $x^N$, random and "
                 "shortest-path deployments; worst case: success against $x^N$ of an attacker who "
                 "observes it and best-responds.",
                 "tab:blind",
                 ["$L(x^N,\\pi^N)$", "$x^N$", "Random", "Shortest path", "Worst case"], data)


def dirichlet_table():
    rows = {g: load("dirichlet", g) for g in GRAPHS}
    r0 = rows["1"][0]
    data = [(GRAPHS[g]["name"],
             [[pm(r["V_optimal"], r["V_optimal_ci"]), dep(r["placement_optimal"]), f(r["V_stackelberg"]),
               pm(r["gain"], r["gain_ci"]), f(r["V_random_avg"]), f(r["V_shortest"]),
               f(r["V_worst_case"]), f(r["V_St_value"])] for r in rows[g]])
            for g in GRAPHS]
    return table(f"Dirichlet regime ($\\rho={r0['lam']}$, $M={r0['M']}$): $J_\\alpha$ estimated on "
                 f"$K'={r0['K_eval']}$ fresh beliefs (seed {r0['seed_eval']}), $x^{{\\mathrm{{Dir}}}}$ "
                 f"computed from $K={r0['K']}$ beliefs (seed {r0['seed_train']}); $\\pm$: 95\\% "
                 "confidence half-width, paired for the gain $J_\\alpha(x^{\\mathrm{St}})-"
                 "J_\\alpha(x^{\\mathrm{Dir}})$. Worst case: success against $x^{\\mathrm{Dir}}$ of an "
                 "attacker who observes it, to be compared with $V^{\\mathrm{St}}$.",
                 "tab:dirichlet",
                 ["$J_\\alpha(x^{\\mathrm{Dir}})$", "$x^{\\mathrm{Dir}}$", "$J_\\alpha(x^{\\mathrm{St}})$",
                  "Gain", "Random", "Shortest path", "Worst case", "$V^{\\mathrm{St}}$"], data)


def runtime_table():
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Instance sizes and largest runtime over $h=1,\\dots,5$ (seconds). "
           "Attacker: computation of one belief-optimal policy (sampling and backward induction).}",
           "\\label{tab:runtime}", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "Graph & $|V|$ & $|E|$ & $|S|$ & Stackelberg & Blind & Dirichlet & Attacker \\\\",
           "\\midrule"]
    for g in GRAPHS:
        G = Graph_choice(g)
        mx = lambda sc, col: max(float(r[col]) for r in load(sc, g))
        out.append(f"{G.name} & {len(G.nodes)} & {len(G.edges)} & {len(G.statespace)} & "
                   f"{f(mx('stackelberg', 'milp_runtime_s'), 3)} & "
                   f"{f(mx('uninformed', 'defender_runtime_s'), 3)} & "
                   f"{f(mx('dirichlet', 'milp_runtime_s'), 3)} & "
                   f"{mx('dirichlet', 'attacker_runtime_s'):.1e} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def sensitivity_table(cases=(("MARA", 1), ("MARA", 2), ("Unguard", 1), ("Unguard", 2), ("Unguard", 3))):
                             #("Unguard", 4), ("Unguard", 5))):
    """Experiment B1: gain / price of x^Dir for each (lambda, M); needs sensitivity_grid.csv."""
    with open(RESULTS_DIR / "sensitivity_grid.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cell = {(r["graph"], int(r["h"]), float(r["lam"]), int(r["M"])): r for r in rows}
    settings = sorted({(float(r["lam"]), int(r["M"])) for r in rows})
    out = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
           "\\caption{Sensitivity of the Dirichlet regime to the reconnaissance quality $\\rho$ and "
           "the concentration $M$: gain $J_\\alpha(x^{\\mathrm{St}})-J_\\alpha(x^{\\mathrm{Dir}})$ "
           "and price $\\max_\\pi L(x^{\\mathrm{Dir}},\\pi)-V^{\\mathrm{St}}$ (paired 95\\% half-width of the "
           f"gain $\\le$ {max(float(r['gain_ci']) for r in rows):.4f}). MiR100: a single attacker policy in "
           "every setting, gain and price are 0.}",
           "\\label{tab:sensitivity}", "\\begin{tabular}{rr" + "c" * len(cases) + "}", "\\toprule",
           "$\\rho$ & $M$ & " + " & ".join(f"{g} $h={h}$" for g, h in cases) + " \\\\", "\\midrule"]
    for title, val in (("Gain", lambda r: float(r["gain"])),
                       ("Price", lambda r: float(r["worst_case"]) - float(r["V_St"]))):
        out.append(f"\\multicolumn{{{len(cases) + 2}}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for lam, M in settings:
            out.append(f"{lam} & {M} & " + " & ".join(f(val(cell[g, h, lam, M])) for g, h in cases)
                       + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def k_table(cases = (("MARA", 1), ("MARA", 2), ("Unguard", 1), ("Unguard", 2), ("Unguard", 3),
                   ("Unguard", 4), ("Unguard", 5))):
    """Experiment B2: out-of-sample J(x^Dir) vs K (mean, spread over seeds); sensitivity_K.csv."""
    with open(RESULTS_DIR / "sensitivity_K.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    Ks = sorted({int(r["K"]) for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    out = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
           "\\caption{Convergence in the number $K$ of training beliefs ($\\rho=0.5$, $M=10$): "
           f"out-of-sample $J_\\alpha(x^{{\\mathrm{{Dir}}}})$ on a fixed set of $K'=2000$ beliefs, mean "
           f"(max $-$ min) over {n_seeds} training seeds, and largest MILP time (s).}}",
           "\\label{tab:K}", "\\begin{tabular}{r" + "c" * len(cases) + "r}", "\\toprule",
           "$K$ & " + " & ".join(f"{g} $h={h}$" for g, h in cases) + " & Time \\\\", "\\midrule"]
    for K in Ks:
        cells = []
        for g, h in cases:
            js = [float(r["J_out_of_sample"]) for r in rows
                  if r["graph"] == g and int(r["h"]) == h and int(r["K"]) == K]
            cells.append(f"{f(sum(js) / len(js))} ({f(max(js) - min(js))})")
        t = max(float(r["runtime_s"]) for r in rows if int(r["K"]) == K)
        out.append(f"{K} & " + " & ".join(cells) + f" & {t:.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def baselines_table():
    """Experiment D: optimal deployment vs four baselines in the three regimes; baselines.csv."""
    with open(RESULTS_DIR / "baselines.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    val = {(r["regime"], r["graph"], int(r["h"]), r["method"]): r["value"] for r in rows}
    methods = ("optimal", "shortest_path", "betweenness", "min_cut", "greedy")
    regimes = (("stackelberg", "Stackelberg"), ("blind", "Blind"), ("dirichlet", "Dirichlet"))
    graphs = [GRAPHS[g]["name"] for g in GRAPHS]
    hs = sorted({int(r["h"]) for r in rows})
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Attacker success under the optimal deployment and four baselines (ties broken by "
           "node index). Betweenness: top-$h$ nodes by directed betweenness centrality; min cut: "
           "minimum node cut between entries and targets, completed or truncated to $h$ nodes by "
           "betweenness; greedy: adds the node that most decreases the regime's objective. "
           "Dirichlet: $\\rho=0.5$, $M=10$, evaluated on $K'=2000$ fresh beliefs.}",
           "\\label{tab:baselines}", "\\begin{tabular}{llccccc}", "\\toprule",
           "Graph & $h$ & Optimal & Shortest path & Betweenness & Min cut & Greedy \\\\"]
    for regime, title in regimes:
        out += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for k, g in enumerate(graphs):
            for i, h in enumerate(hs):
                out.append(f"{g if i == 0 else ''} & {h} & "
                           + " & ".join(f(val[regime, g, h, m]) for m in methods) + " \\\\")
            if k < len(graphs) - 1:
                out.append("\\cmidrule(lr){1-7}")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def bracketing_table():
    """CTR 2023 comparison: V^Blind <= V^NF <= V^St; ctr2023_bracketing.csv."""
    with open(RESULTS_DIR / "ctr2023_bracketing.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{The normal-form game of \\cite{rass2023} solved on the present model "
           "($V^{\\mathrm{NF}}$: the attacker knows the defender's mixed strategy, not the deployment "
           "drawn), between the blind ($V^{\\mathrm{Blind}}$) and Stackelberg ($V^{\\mathrm{St}}$) "
           "regimes. Support: number of deployments in the equilibrium mixture.}",
           "\\label{tab:ctr2023}", "\\begin{tabular}{llcccc}", "\\toprule",
           "Graph & $h$ & $V^{\\mathrm{Blind}}$ & $V^{\\mathrm{NF}}$ & $V^{\\mathrm{St}}$ & Support \\\\",
           "\\midrule"]
    names = [r["graph"] for r in rows]
    for i, r in enumerate(rows):
        first = i == 0 or names[i - 1] != r["graph"]
        if first and i > 0:
            out.append("\\cmidrule(lr){1-6}")
        support = len(ast.literal_eval(r["equilibrium_mixture"]))
        out.append(f"{r['graph'] if first else ''} & {r['h']} & {f(r['V_blind'])} & "
                   f"{f(r['V_normal_form'])} & {f(r['V_stackelberg'])} & {support} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    path = RESULTS_DIR / "tables.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by make_tables.py from csv_results/*.csv -- do not edit by hand.\n")
        parts = [stackelberg_table(), blind_table(), dirichlet_table(), runtime_table()]
        if (RESULTS_DIR / "sensitivity_grid.csv").exists():
            parts.append(sensitivity_table())
        if (RESULTS_DIR / "sensitivity_K.csv").exists():
            parts.append(k_table())
        if (RESULTS_DIR / "baselines.csv").exists():
            parts.append(baselines_table())
        if (RESULTS_DIR / "ctr2023_bracketing.csv").exists():
            parts.append(bracketing_table())
        fh.write("\n".join(parts))
    print(f"  [IO] tables saved -> {path}")