import numpy as onp

from stellacode import np
import typing as tp


class RotateNTimes:
    def __init__(self, angle: float, max_num: int = 1, min_num: int = 0):
        """
        Rotate and duplicate a surface along the toroidal angle.

        Args:
            * angle: angle of each rotation.
            * max_num: maximum number of rotations.
            * min_num: minimum number of rotations.
        """
        self.min_num = min_num
        self.max_num = max_num
        self.angle = angle
        rot = onp.array(
            [
                [onp.cos(angle), -onp.sin(angle), 0],
                [onp.sin(angle), onp.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        self.rot_ten = np.stack([onp.linalg.matrix_power(rot, i) for i in range(self.min_num, self.max_num)])

    @classmethod
    def from_nfp(cls, nfp: int):
        return cls(2 * np.pi / nfp, nfp)

    def __call__(self, ten, stack_dim: tp.Optional[int] = None):
        """
        tensor dimensions are always: poloidal x toroidal x 3 x others
        """
        if stack_dim is not None:
            return np.concatenate([ten] * self.n_rot, axis=stack_dim)
        elif len(ten.shape) == 2:
            return np.concatenate([ten] * self.n_rot, axis=1)
        elif len(ten.shape) == 3:
            assert ten.shape[2] == 3
            return np.reshape(
                np.einsum("opq,ijq->iojp", self.rot_ten, ten),
                (ten.shape[0], -1, 3),
            )
        elif len(ten.shape) == 4:
            assert ten.shape[2] == 3
            return np.reshape(
                np.einsum("opq,ijqa->iojpa", self.rot_ten, ten),
                (ten.shape[0], -1, 3, ten.shape[-1]),
            )
        elif len(ten.shape) == 5:
            assert ten.shape[2] == 3
            return np.reshape(
                np.einsum("opq,ijqab->iojpab", self.rot_ten, ten),
                (ten.shape[0], -1, 3, ten.shape[-2], ten.shape[-1]),
            )
        else:
            raise NotImplementedError

    @property
    def n_rot(self):
        return self.max_num - self.min_num
