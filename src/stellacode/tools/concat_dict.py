import typing as tp

from pydantic import BaseModel

from stellacode import np


class ConcatDictArray:
    """
    Concatenate and deconcatenate a dict of arrays in and from a vector
    """

    def apply(self, darr):
        """
        Concatenate a dict of arrays into a vector.

        Args:
            darr (dict): The dictionary of arrays.

        Returns:
            np.ndarray: The concatenated array.
        """
        # Get the shape of each array
        self.shapes = {k: np.array(np.array(v).shape, dtype=int)
                       for k, v in darr.items()}
        # Reshape each array into a 1D array and concatenate them
        return np.concatenate([np.reshape(v, -1) for v in darr.values()])

    def unapply(self, arr):
        """
        Deconcatenate a concatenated dict of arrays.

        Args:
            arr (np.ndarray): The concatenated array.

        Returns:
            dict: The dict of arrays.
        """
        # Check that concatenate was called before
        assert "shapes" in dir(self), "You should call concat first."

        ind = 0

        darr = {}
        # Iterate over the concatenated dict of arrays
        for k, sh in self.shapes.items():
            # Calculate the index of the next array
            new_ind = ind + np.prod(sh)
            # Reshape the array and add it to the dict
            darr[k] = np.reshape(arr[ind:new_ind], sh)
            # Update the index
            ind = new_ind

        return darr


ScaleDict = tp.Dict[str, tp.Union[float, tp.Tuple[float, float], None]]


class ScaleDictArray(BaseModel):
    """
    Scale a dict of arrays to have a 0 average and a unit std.
    Used in the Optimizer class.

    Args:
        * scales: stored list of scales, scales which are preset will not be computed from the arrays
        * min_std: minimum possible std (to avoid division by zero errors)
        * additional_scale: scale all arrays by this value
    """

    scales: ScaleDict = {}
    min_std: float = 1e-8
    additional_scale: float = 1

    model_config = dict(arbitrary_types_allowed=True)

    def apply(self, darr):
        """
        Scale a dictionary of arrays to have a 0 mean and a unit std.

        Args:
            darr (dict): dictionary of arrays

        Returns:
            scaled_darr (dict): dictionary of scaled arrays
        """
        for k, v in darr.items():
            # If the scale is already defined, skip the computation
            if k not in self.scales.keys():
                # If the array is a scalar, set the scale to the array itself
                if len(np.array(v).shape) == 0:
                    self.scales[k] = v
                # If the array is an array, compute the mean and the std
                else:
                    mean_ = np.mean(v)
                    self.scales[k] = mean_, max(
                        np.linalg.norm(v - mean_), self.min_std)

        scaled_darr = {}
        for k, v in darr.items():
            # If the scale is in the scales dict
            if k in self.scales:
                # If the scale is None, pass
                if self.scales[k] is None:
                    pass
                # If the array is a scalar, scale it
                elif len(np.array(v).shape) == 0:
                    scaled_darr[k] = v / self.scales[k] * self.additional_scale
                # If the array is an array, scale it
                else:
                    mean_, std = self.scales[k]
                    scaled_darr[k] = (v - mean_) / std * self.additional_scale

        return scaled_darr

    def unapply(self, scaled_darr):
        """
        Scale all arrays back to their original state.

        Args:
            scaled_darr (dict): dictionary of scaled arrays

        Returns:
            darr (dict): dictionary of unscaled arrays
        """
        darr = {}
        # Iterate over all arrays in the dictionary
        for k, v in scaled_darr.items():
            # Check if the array has a scale
            if k in self.scales:
                # If the scale is None, do nothing
                if self.scales[k] is None:
                    pass
                # If the array is a scalar, unscale it
                elif len(np.array(v).shape) == 0:
                    darr[k] = v * self.scales[k] / self.additional_scale
                # If the array is an array, unscale it
                else:
                    # Get the mean and std of the scale
                    mean_, std = self.scales[k]
                    # Unscale the array
                    darr[k] = std * v / self.additional_scale + mean_

        return darr


class ConcatScaleDictArray(BaseModel):
    """
    Concat and scale a dict of arrays
    """

    concater: ConcatDictArray = ConcatDictArray()
    scaler: tp.Optional[ScaleDictArray] = None

    model_config = dict(arbitrary_types_allowed=True)

    def apply(self, darr):
        """
        Apply the scaler if it exists, and then concatenate the dictionary of arrays

        Args:
            darr (dict): dictionary of arrays

        Returns:
            numpy.ndarray: concatenated array
        """
        # If a scaler is defined, apply it to the dictionary of arrays
        if self.scaler is not None:
            darr = self.scaler.apply(darr)

        # Concatenate the dictionary of arrays
        return self.concater.apply(darr)

    def unapply(self, arr):
        """
        Unapply the scaler if it exists, and then deconcatenate the array into a dictionary of arrays

        Args:
            arr (numpy.ndarray): concatenated array

        Returns:
            dict: dictionary of arrays
        """
        # Deconcatenate the array into a dictionary of arrays
        darr = self.concater.unapply(arr)

        # If a scaler is defined, unapply it to the dictionary of arrays
        if self.scaler is not None:
            darr = self.scaler.unapply(darr)

        return darr
