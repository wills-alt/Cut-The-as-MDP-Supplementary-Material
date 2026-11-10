"""Perceived detection: h sensors placed one by one, without replacement,
each drawn proportionally to p among the nodes not yet chosen.

Same method as the hand calculation "par couples":
  1. compute the probability of every set of chosen nodes,
  2. p_tilde(v) = sum of the probabilities of the sets that contain v.
"""


def inclusion_probabilities(p, h):
    nodes = [v for v in p if p[v] > 0]
    if len(nodes) <= h:                      # all nodes of the support are taken
        return {v: (1.0 if p[v] > 0 else 0.0) for v in p}

    # sets[S] = probability that the sensors placed so far occupy exactly the set S
    sets = {frozenset(): 1.0}  # empty set with probability 1
    for _ in range(h):                  
        new_sets = {}
        for S, prob_S in sets.items():
            weight_left = 1.0 - sum(p[u] for u in S)       # total weight of remaining nodes
            for v in nodes:
                if v not in S:                           
                    T = S | {v}
                    new_sets[T] = new_sets.get(T, 0.0) + prob_S * p[v] / weight_left
        sets = new_sets

    # p_tilde(v) = Pr(v is in the chosen set)
    p_tilde = {v: 0.0 for v in p}
    for S, prob_S in sets.items():
        for v in S:
            p_tilde[v] += prob_S
    return p_tilde


if __name__ == "__main__":
    print(inclusion_probabilities({1: .6, 2: .3, 3: .1}, 2))   # 0.924, 0.783, 0.293
    print(inclusion_probabilities({1: .6, 2: .4, 3: 0.}, 2))   # 1, 1, 0