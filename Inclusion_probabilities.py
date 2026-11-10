"""Perceived detection under 'h sensors placed without replacement'.

For a belief p (a point of the simplex), h sensors are placed one after the
other; each is drawn proportionally to p among the nodes not yet chosen.
inclusion_probabilities(p, h)[v] = Pr(v receives a sensor | p).

Properties: each value in [0, 1], the values sum to h, no cap is needed.
If p has at most h positive entries (e.g. rho = 1), these nodes are all chosen.

Exact computation (successive sampling = exponential race): T_v ~ Exp(p(v))
independent, the chosen set is the h smallest T_v, hence
    pi(v) = int_0^inf p(v) exp(-p(v) t) Pr(N_{-v}(t) <= h-1) dt,
where N_{-v}(t) = #{u != v : T_u < t} (Poisson-binomial, computed by DP).
Checked against brute-force enumeration of ordered placements (error < 1e-15).
"""
import numpy as np
from scipy.integrate import quad_vec


def inclusion_probabilities(p, h):
    keys = list(p.keys())  
    w = np.array([p[k] for k in keys], dtype=float)
    pos = np.flatnonzero(w > 0)  # give the support of the vector w (positions with non null values)
    out = np.zeros(len(w))
    if len(pos) <= h:                       # support of size <= h: all taken
        out[pos] = 1.0
        return dict(zip(keys, out))    # Dict with 1.0 on suppport and 0 otherwise
    wp = w[pos]
    n = len(wp)
    eye = np.eye(n, dtype=bool)

    def integrand(t):     
        F = -np.expm1(-wp * t)              # Pr(T_u < t) =  - (e^(-wp) - 1).
        dist = np.zeros((n, h))             # row i: law of N_{-i}(t), truncated at h-1
        dist[:, 0] = 1.0
        for j in range(n):
            f = np.where(eye[:, j], 0.0, F[j])[:, None] # Pr(T_u < t) if 0.0 and 0.0 if 1.0
            for i in range(n):  # We are looking for the probability that at most h-1 nodes arrived before i
                for k in range(h - 1, 0, -1):
                    dist[i, k] = dist[i, k] * (1 - f[i, 0]) + dist[i, k - 1] * f[i, 0]   # Probability that k nodes reached before t
                dist[i, 0] = dist[i, 0] * (1 - f[i, 0])  # Prob that non node reached before t
        return wp * np.exp(-wp * t) * dist.sum(axis=1)

    val, _ = quad_vec(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
    out[pos] = val
    return dict(zip(keys, out))


if __name__ == "__main__":
    print(inclusion_probabilities({1: .6, 2: .3, 3: .1}, 2))   # (0.9238, 0.7833, 0.2929)
    print(inclusion_probabilities({1: .6, 2: .4, 3: 0.}, 2))   # (1, 1, 0)