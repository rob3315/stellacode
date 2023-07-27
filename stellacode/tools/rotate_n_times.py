import numpy as np


class RotateNTimes:
    def __init__(self, num_tor_symmetry):
        self.num_tor_symmetry = num_tor_symmetry
        rot = np.array(
            [
                [np.cos(2 * np.pi / num_tor_symmetry), -np.sin(2 * np.pi / num_tor_symmetry), 0],
                [np.sin(2 * np.pi / num_tor_symmetry), np.cos(2 * np.pi / num_tor_symmetry), 0],
                [0, 0, 1],
            ]
        )
        self.rot_ten = np.stack([np.linalg.matrix_power(rot, i) for i in range(num_tor_symmetry)])

    def call(self, ten):
        """
        tensor dimensions are always: poloidal x toroidal x 3 x others
        """
        if len(ten.shape) == 2:
            return np.concatenate([ten] * self.num_tor_symmetry, axis=1)
        elif len(ten.shape) == 3:
            assert ten.shape[-1] == 3
            return np.reshape(
                np.einsum("opq,ijq->iojp", self.rot_ten, ten),
                (ten.shape[0], -1, 3),
            )
        elif len(ten.shape) == 4:
            assert ten.shape[-1] == 3
            return np.reshape(
                np.einsum("opq,ijqa->iojpa", self.rot_ten, ten),
                (ten.shape[0], -1, 3),
            )
        else:
            raise NotImplementedError
