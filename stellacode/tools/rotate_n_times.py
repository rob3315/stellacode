import numpy as onp

from stellacode import np


class RotateNTimes:
    def __init__(self, angle: float, number: int = 1):
        self.number = number
        self.angle = angle
        rot = onp.array(
            [
                [onp.cos(angle), -onp.sin(angle), 0],
                [onp.sin(angle), onp.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        self.rot_ten = np.stack([onp.linalg.matrix_power(rot, i) for i in range(number)])

    @classmethod
    def from_nfp(cls, nfp: int):
        return cls(2 * np.pi / nfp, nfp)

    def __call__(self, ten):
        """
        tensor dimensions are always: poloidal x toroidal x 3 x others
        """
        if len(ten.shape) == 2:
            return np.concatenate([ten] * self.number, axis=1)
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
