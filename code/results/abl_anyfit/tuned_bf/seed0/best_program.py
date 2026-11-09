# eval 131, train excess 0.03965
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=0, hi=22, penalty=99): the tightest bin wins, but leaving a gap
    strictly between 0 and 22 units costs 99. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 99 * ((r > 0) & (r < 22))
