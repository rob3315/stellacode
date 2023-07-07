from stellacode import np


class ConcatDictArray:
    def concat(self, darr):
        self.shapes = {k: np.array(np.array(v).shape, dtype=int) for k, v in darr.items()}
        return np.concatenate([np.reshape(v, -1) for v in darr.values()])

    def unconcat(self, arr):
        assert "shapes" in dir(self), "You should call concat first."
        ind = 0

        darr = {}
        for k, sh in self.shapes.items():
            new_ind = ind + np.prod(sh)
            darr[k] = np.reshape(arr[ind:new_ind], sh)
            ind = new_ind

        return darr
