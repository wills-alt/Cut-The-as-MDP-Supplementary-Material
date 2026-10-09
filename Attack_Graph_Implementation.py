# Attack_Graph_Implementation.py: implementation of transition functions, graphs studied ...

from collections import deque
from scipy.stats import poisson

class AttackGraph:
    """
    Attack graph with augmented state (node, steps_taken) and absorbing states:
      - Sink: ("Sink", None)
      - Stop: defender idle period over
      - Targets: (target_name, None) for each target in target_nodes

    Supports:
      - attacker_entry_nodes: the true starting nodes the attacker may use.
      - defender_belief_nodes: the nodes the defender believes are possible starts.
      - build_initial_distributions() producing attacker_init & defender_init arrays.
    """

    def __init__(self, nodes, edges,
                 entry_nodes,       # true attacker possible starts (list of node names)
                 target_nodes, lambda_attack,
                 lambda_defender,    # list of target node names (one or more)
                 start_nodes = None, # defender belief 
                 defense_pattern = "exponential"):
        # Graph structure
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.target_nodes = (list(target_nodes)
                             if isinstance(target_nodes, (list, tuple, set))
                             else [target_nodes])
        self.entry_nodes = list(entry_nodes)
        self.adjacency = self._build_adjacency()
        self._check_acyclic()

        # Support of nu (uniform start distribution): V \ F by default
        self.start_nodes = (list(start_nodes) if start_nodes is not None
                            else [n for n in self.nodes if n not in self.target_nodes])

        # Protectable nodes: V_spot = V \ (F u entries); these are the nodes that the defender can access
        self.V_spot = [n for n in self.nodes
                       if n not in self.target_nodes and n not in self.entry_nodes]
        
        # Timing of the actions
        self.lambda_attack = lambda_attack
        self.lambda_defender = lambda_defender

        # Absorbing states
        self.SINK = ("Sink", None)     # detection
        self.STOP = ("Stop", None)     # defender's idle period ended
        self.TARGETS = [(t, None) for t in self.target_nodes]  # TARGETS list of tuples (name, None)

        self.c_max = None
        self.statespace = None
        self.P = None

        #defense pattern
        self.defense_pattern = defense_pattern

    def _check_acyclic(self):
        """Kahn's algorithm: raise if G contains a cycle (our model requires a DAG, to ensure finiteness)."""
        indeg = {n: 0 for n in self.nodes}
        for _, t in self.edges:
            indeg[t] += 1
        queue = deque(n for n in self.nodes if indeg[n] == 0)
        seen = 0
        while queue:
            n = queue.popleft(); seen += 1
            for m in self.adjacency[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen != len(self.nodes):
            raise ValueError("Attack graph contains a cycle; RED FLAG since the model requires a DAG.")
        
    def _build_adjacency(self):
        adj = {n: [] for n in self.nodes}
        for s, t in self.edges:
            if s not in adj or t not in adj:
                raise ValueError(f"Edge ({s},{t}) contains unknown node, big issue to verify.")
            adj[s].append(t)
        return adj

    def step_probability(self, c):
        """q_c = Pr(N >= c+1 | N >= c), the probability that the attacker completes one more move given c moves already made in the current idle period (Eq. 1).
          'exponential': N geometric, Pr(N >= k) = r^k  ->  q_c = r = lam/(lam + lam_D)
          'periodic'   : N ~ Poisson(lam),  Pr(N >= k) = poisson.sf(k-1, lam)
        """
        lam, lam_D = self.lambda_attack, self.lambda_defender
        if self.defense_pattern == "exponential":
            return lam / (lam + lam_D)
        if self.defense_pattern == "periodic":
            S_c  = poisson.sf(c - 1, lam)     # Pr(N >= c)
            S_c1 = poisson.sf(c, lam)         # Pr(N >= c+1)
            return S_c1 / S_c if S_c > 0 else 0.0
        raise ValueError(f"Unknown defense pattern: {self.defense_pattern}")    

    def build_statespace(self):
        """ Reachable augmented states (Definition 1): all (node, c) such that node can be reached in exactly c edges from a start node without 
        passing through a target, plus the absorbing states. Also sets c_max = longest such path.
        """
        lengths = {n: set() for n in self.nodes}
        queue = deque()
        for n in self.start_nodes:
            lengths[n].add(0)
            queue.append((n, 0))
        while queue:
            node, c = queue.popleft()
            if node in self.target_nodes:          # targets are absorbing
                continue
            for nxt in self.adjacency[node]:
                if c + 1 not in lengths[nxt]:
                    lengths[nxt].add(c + 1)
                    queue.append((nxt, c + 1))

        states = [(n, c) for n in self.nodes if n not in self.target_nodes
                  for c in sorted(lengths[n])]
        self.c_max = max(c for _, c in states)
        self.statespace = states + self.TARGETS + [self.SINK, self.STOP]
        return self.statespace


    def get_transition_function(self):
        """ Base kernel P (Equation. 1), without protections: P[s][a] = {s': prob}.
        From s = (i, c), action goto_j:
            -> (j, c+1) or target state (j, None)   with prob q_c
            -> STOP                                 with prob 1 - q_c
        """
        if self.statespace is None:
            self.build_statespace()
        if self.P is not None:
            return self.P

        S = set(self.statespace)
        absorbing = set(self.TARGETS) | {self.SINK, self.STOP}
        P = {}
        for s in self.statespace:
            if s in absorbing:
                P[s] = {'stay': {s: 1.0}}
                continue
            node, c = s
            if not self.adjacency[node]:
                raise ValueError(f"Non-target node {node} has no outgoing edge.")
            q = self.step_probability(c)
            P[s] = {}
            for nxt in self.adjacency[node]:
                succ = (nxt, None) if nxt in self.target_nodes else (nxt, c + 1)
                if succ not in S:
                    raise ValueError(f"Successor {succ} of {s} missing from the state space. There is an error in state space computation.")
                row = {succ: q, self.STOP: 1.0 - q}
                P[s][f"goto_{nxt}"] = {t: p for t, p in row.items() if p > 0}

        for s, acts in P.items():                      # sanity check
            for a, row in acts.items():
                if abs(sum(row.values()) - 1.0) > 1e-12:
                    raise ValueError(f"Row {s}, {a} does not sum to 1.")
        self.P = P
        return P

    def belief_transition(self, q):
            """ Belief-modified kernel P^q (Sections IV-V): the attacker believes that leaving node i is detected with probability q[i].
                P^q(SINK | s=(i,c), a) = q[i]
                P^q(s'   | s=(i,c), a) = (1 - q[i]) * P(s' | s, a)      for the other s'
            Used with q = blind belief (h/|V_spot|) or a Dirichlet sample.
            """
            P = self.get_transition_function()
            Pq = {}
            for s, acts in P.items():
                if s[1] is None:                       # absorbing states unchanged
                    Pq[s] = acts
                    continue
                qi = q.get(s[0], 0.0)
                Pq[s] = {}
                for a, row in acts.items():
                    new_row = {t: (1.0 - qi) * p for t, p in row.items()}
                    if qi > 0:
                        new_row[self.SINK] = qi
                    Pq[s][a] = new_row
            return Pq
    


    def get_reward(self, state):
        """R(s) = 1 at target states, 0 elsewhere (SINK, STOP, transient states)."""
        return 1.0 if state in self.TARGETS else 0.0



# ===============================
# Case-study graphs (Section VI).
# ===============================

GRAPHS = {
    "1": dict(  # MARA robotic arm
        name="MARA",
        nodes=list(range(1, 10)),
        edges=[(1, 2), (2, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 7), (7, 8), (8, 9)],
        entry_nodes=[1],
        target_nodes=[6, 9],
    ),
    "2": dict(  # MiR100 mobile robot
        name="MiR100",
        nodes=list(range(1, 17)),
        edges=[(1, 5), (2, 9), (2, 11), (2, 10), (2, 16),
               (3, 6), (3, 8), (4, 7),
               (5, 15), (6, 8), (7, 10), (7, 14), (7, 16),
               (8, 10), (8, 14), (8, 16),
               (9, 14), (10, 15), (11, 13), (11, 14), (11, 16),
               (15, 12), (15, 13), (15, 16)],
        entry_nodes=[1, 2, 3, 4],
        target_nodes=[12, 13, 14, 16],
    ),
    "3": dict(  # Unguard microservice network (by Dynatrace)
        name="Unguard",
        nodes=list(range(1, 34)),
        edges=[
            # Path 1: Redis
            (1, 6), (6, 11), (6, 12), (11, 16), (12, 16), (11, 17), (12, 17),
            (16, 23), (17, 24), (23, 29), (24, 30), (23, 10), (24, 10),
            (10, 5), (10, 7), (10, 8), (10, 9),
            # Path 2: SQL injection
            (1, 2), (1, 3), (1, 4), (2, 13), (3, 13), (4, 13), (13, 18), (18, 5),
            (5, 25), (25, 30), (25, 31),
            # Path 3: Envoy bypass
            (3, 14), (4, 14), (14, 19), (19, 26), (26, 7), (26, 8), (26, 9), (26, 10),
            (7, 32), (8, 32), (9, 32), (10, 32),
            # Path 4: SSRF
            (2, 15), (3, 15), (15, 20), (15, 21), (15, 22), (20, 27), (21, 5), (21, 6),
            (22, 28), (27, 33), (28, 33), (5, 33), (6, 33),
            # Edges out of targets (ignored: targets are absorbing)
            (29, 33), (30, 33), (31, 33), (32, 33),
        ],
        entry_nodes=[1],
        target_nodes=[29, 30, 31, 32, 33],
    ),
}


def Graph_choice(graph_name, lambda_attack = 2, lambda_defender = 1, defense_pattern="exponential"):
    """Build the AttackGraph of a case study, with state space and base kernel P."""
    if graph_name not in GRAPHS:
        raise ValueError(f"Unknown graph: {graph_name}")
    d = GRAPHS[graph_name]
    G = AttackGraph(d["nodes"], d["edges"], d["entry_nodes"], d["target_nodes"],
                    lambda_attack, lambda_defender, defense_pattern = defense_pattern)
    G.name = d["name"]
    G.get_transition_function()
    return G


if __name__ == "__main__":
    for key in GRAPHS:
        G = Graph_choice(key)
        print(f"{G.name}: |V|={len(G.nodes)} |E|={len(G.edges)} |S|={len(G.statespace)} "
              f"c_max={G.c_max} V_spot={G.V_spot}")
        print("the transition state is", G.P)


