from pydantic import BaseModel

from stellacode import np


class Constraint(BaseModel):
    limit: float = 0.0
    distance: float = 1.0
    weight: float = 1.0
    minimum: bool = True
    method: str = "inverse"

    def constraint(self, val):
        if self.minimum:
            return val - self.limit
        else:
            return self.limit - val

    def barrier(self, val):
        ctr = self.constraint(val)
        if self.method == "quadratic":
            return (np.maximum(-ctr, 0) / self.distance) ** 2

        clipped_dist = np.maximum(self.distance - ctr, 0)
        if self.method == "inverse":
            cost = clipped_dist**2 / (1 - clipped_dist / self.distance)
        elif self.method == "quadratic_log":
            cost = np.log(np.clip(ctr / self.distance, 0, 1)) ** 2
        else:
            raise NotImplementedError
        return np.where(
            ctr > 0.0,
            self.weight * cost,
            np.inf,
        )


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


def merge_dataclasses(dc1, dc2):
    assert type(dc1) == type(dc2)
    for field_name in type(dc1).__fields__:
        setattr(dc1, field_name, getattr(dc2, field_name))
    return dc1
