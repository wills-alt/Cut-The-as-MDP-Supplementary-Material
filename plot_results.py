"""Draw Fig. 3-5 (one figure per graph) from csv_results/*.csv."""
from results_io import plot_scenarios_side_by_side
from Attack_Graph_Implementation import GRAPHS

if __name__ == "__main__":
    for g in GRAPHS:
        plot_scenarios_side_by_side(g)