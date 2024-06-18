import typing as tp

import numpy as onp

from stellacode import np


class RotateNTimes:
    def __init__(self, angle: float, max_num: int = 1, min_num: int = 0):
        """
        Rotate and duplicate a surface along the toroidal angle.

        Args:
            angle (float): angle of each rotation.
            max_num (int, optional): maximum number of rotations. Defaults to 1.
            min_num (int, optional): minimum number of rotations. Defaults to 0.
        """
        # Initialize the minimum and maximum number of rotations
        self.min_num = min_num
        self.max_num = max_num
        # Initialize the angle of each rotation
        self.angle = angle
        # Compute the rotation matrix
        rot = onp.array(
            [
                [onp.cos(angle), -onp.sin(angle), 0],
                [onp.sin(angle), onp.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        # Compute the rotation tensors for each rotation
        self.rot_ten = np.stack(
            [onp.linalg.matrix_power(rot, i)
             for i in range(self.min_num, self.max_num)]
        )

    @classmethod
    def from_nfp(cls, nfp: int):
        """
        Factory method to create a RotateNTimes object from the number of field periods.

        Args:
            nfp (int): Number of field periods.

        Returns:
            RotateNTimes: A RotateNTimes object that rotates a surface along the toroidal angle
                `2 * np.pi / nfp` a number of times equal to the number of field periods.
        """
        # Compute the angle of each rotation
        angle = 2 * np.pi / nfp
        # Return a RotateNTimes object with the given angle and number of field periods
        return cls(angle, nfp)

    def __call__(self, ten, stack_dim: tp.Optional[int] = None):
        """
        Rotate a tensor `n_rot` times.

        Args:
            ten (ArrayLike): The tensor to be rotated.
            stack_dim (int, optional): The dimension along which to stack the
                rotated tensors. Defaults to None.

        Returns:
            ArrayLike: The rotated tensor.

        Raises:
            NotImplementedError: If the tensor has more than 5 dimensions.

        Note:
            The tensor dimensions are always: poloidal x toroidal x 3 x others.
        """
        # If stack_dim is specified, concatenate the tensor `n_rot` times
        if stack_dim is not None:
            return np.concatenate([ten] * self.n_rot, axis=stack_dim)
        # If the tensor has 2 dimensions, stack them horizontally `n_rot` times
        elif len(ten.shape) == 2:
            return np.concatenate([ten] * self.n_rot, axis=1)
        # If the tensor has 3 dimensions and the third dimension is size 3,
        # apply the rotation matrix `n_rot` times
        elif len(ten.shape) == 3:
            # Assert that the third dimension is size 3
            assert ten.shape[2] == 3
            # Apply the rotation matrix and reshape the result
            return np.reshape(
                np.einsum("opq,ijq->iojp", self.rot_ten, ten),
                (ten.shape[0], -1, 3),
            )
        # If the tensor has 4 dimensions and the third dimension is size 3,
        # apply the rotation matrix `n_rot` times
        elif len(ten.shape) == 4:
            # Assert that the third dimension is size 3
            assert ten.shape[2] == 3
            # Apply the rotation matrix and reshape the result
            return np.reshape(
                np.einsum("opq,ijqa->iojpa", self.rot_ten, ten),
                (ten.shape[0], -1, 3, ten.shape[-1]),
            )
        # If the tensor has 5 dimensions and the third dimension is size 3,
        # apply the rotation matrix `n_rot` times
        elif len(ten.shape) == 5:
            # Assert that the third dimension is size 3
            assert ten.shape[2] == 3
            # Apply the rotation matrix and reshape the result
            return np.reshape(
                np.einsum("opq,ijqab->iojpab", self.rot_ten, ten),
                (ten.shape[0], -1, 3, ten.shape[-2], ten.shape[-1]),
            )
        # If the tensor has more than 5 dimensions, raise an error
        else:
            raise NotImplementedError

    @property
    def n_rot(self):
        """
        Returns the number of times the tensor will be rotated.

        This is computed by subtracting the minimum number of rotations from the maximum number of rotations.

        Returns:
            int: The number of times the tensor will be rotated.
        """
        # Compute the number of times the tensor will be rotated
        # by subtracting the minimum number of rotations from the maximum number of rotations
        return self.max_num - self.min_num
