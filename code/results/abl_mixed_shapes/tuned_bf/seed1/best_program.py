# eval 32, train excess 0.03277
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Tuned Best Fit (lo=6, hi=24, penalty=68): the tightest bin wins, but leaving a gap
    strictly between 6 and 24 units costs 68. lo=0, hi=0, penalty=0 is plain Best Fit."""
    r = bins - item                                  # space left in each bin after placing the item
    return -r - 68 * ((r > 6) & (r < 24))
