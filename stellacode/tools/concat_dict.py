import typing as tp

from pydantic import BaseModel

from stellacode import np


class ConcatDictArray:
    """
    Concatenate and deconcatenate a dict of arrays in and from a vector
    """

    def apply(self, darr):
        self.shapes = {k: np.array(np.array(v).shape, dtype=int) for k, v in darr.items()}
        return np.concatenate([np.reshape(v, -1) for v in darr.values()])

    def unapply(self, arr):
        assert "shapes" in dir(self), "You should call concat first."
        ind = 0

        darr = {}
        for k, sh in self.shapes.items():
            new_ind = ind + np.prod(sh)
            darr[k] = np.reshape(arr[ind:new_ind], sh)
            ind = new_ind

        return darr


ScaleDict = tp.Dict[str, tp.Union[float, tp.Tuple[float, float], None]]


class ScaleDictArray(BaseModel):
    """
    Scale a dict of arrays to have a 0 average and a unit std.

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
        for k, v in darr.items():
            if k not in self.scales.keys():
                if len(np.array(v).shape) == 0:
                    self.scales[k] = v
                else:
                    mean_ = np.mean(v)
                    self.scales[k] = mean_, max(np.linalg.norm(v - mean_), self.min_std)

        scaled_darr = {}
        for k, v in darr.items():
            if k in self.scales:
                if self.scales[k] is None:
                    pass
                elif len(np.array(v).shape) == 0:
                    scaled_darr[k] = v / self.scales[k] * self.additional_scale
                else:
                    mean_, std = self.scales[k]
                    scaled_darr[k] = (v - mean_) / std * self.additional_scale

        return scaled_darr

    def unapply(self, scaled_darr):
        darr = {}
        for k, v in scaled_darr.items():
            if k in self.scales:
                if self.scales[k] is None:
                    pass
                elif len(np.array(v).shape) == 0:
                    darr[k] = v * self.scales[k] / self.additional_scale
                else:
                    mean_, std = self.scales[k]
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
        if self.scaler is not None:
            darr = self.scaler.apply(darr)
        return self.concater.apply(darr)

    def unapply(self, arr):
        darr = self.concater.unapply(arr)
        if self.scaler is not None:
            darr = self.scaler.unapply(darr)
        return darr
