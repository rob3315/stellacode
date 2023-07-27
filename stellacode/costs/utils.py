""" contained non linear functions (and there derivative) to impose constraints
"""
from stellacode import np


def inverse_barrier(val, min_val, distance, weight=1.0):
    # infinite for values lower than min_val
    # penalize from min_val+distance
    clipped_dist = np.maximum(min_val + distance - val, 0)
    return np.where(
        val > min_val,
        weight * clipped_dist**2 / (1 - clipped_dist / distance),
        np.inf,
    )


def quadratic_log_barrier(val, min_val, distance, weight=1.0):
    # infinite for values lower than min_val
    # penalize from min_val+distance
    return np.log(np.clip((val - min_val) / distance, 0, 1)) ** 2 * weight


def quadratic_barrier(val, min_val, weight=1.0):
    # penalize from min_val
    return weight * np.maximum(min_val - val, 0) ** 2
