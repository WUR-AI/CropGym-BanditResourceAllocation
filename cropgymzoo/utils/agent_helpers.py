import itertools
import numpy as np

# ---------------------------------------------------------------------
# helper: all integer n-tuples whose entries sum to Q   (“stars-and-bars”)
# ---------------------------------------------------------------------
def make_super_arms(n_fields: int, Q: int):
    """
    Return list of integer vectors (length = n_fields, each >=0) summing to Q.
    Uses the "stars and bars" bijection with combinations_with_replacement.
    """
    # choose Q stars’ positions among (Q+n_fields-1) slots
    bars = itertools.combinations(range(Q + n_fields - 1), n_fields - 1)
    super_arms = []
    for cutpoints in bars:
        # prepend −1, append last slot, take diffs → bucket sizes
        splits = (-1,) + cutpoints + (Q + n_fields - 1,)
        alloc = np.diff(splits) - 1               # vector length n_fields
        super_arms.append(alloc.astype(np.int32))
    return super_arms


def make_super_arms_limited(max_q: list, Q: int):
    """
    max_q : 1-D int array of length n_fields holding per-field maxima (in quanta)
    Q     : total quanta to distribute
    Returns all allocation vectors a with sum == Q and a[j] ≤ max_q[j].
    """
    n = len(max_q)
    super_arms = []

    # recursive back-tracking over fields
    def backtrack(idx, remaining, prefix):
        if idx == n:                      # all fields decided
            if remaining == 0:
                super_arms.append(np.array(prefix, dtype=np.int32))
            return
        max_here = min(max_q[idx], remaining)
        for a in range(max_here + 1):
            backtrack(idx + 1, remaining - a, prefix + [a])

    backtrack(0, Q, [])
    return super_arms