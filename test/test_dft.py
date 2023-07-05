import numpy as onp
import pytest

from stellacode import np
from stellacode.surface.utils import fourier_transform


@pytest.mark.skip("Not working")
def test_dft():
    coefs = onp.random.rand(5, 2)

    fourier_transform_vec = np.vectorize(fourier_transform, excluded=(0,))

    N = 10

    uval = np.arange(N) * 2 * np.pi / N
    dft_val = fourier_transform_vec(coefs, uval)

    c_coefs = np.concatenate(
        (
            (coefs[:, 0] + 1j * coefs[:, 1]) / 2,
            np.array([0.0]),
            (coefs[:, 0] - 1j * coefs[:, 1])[::-1] / 2,
        )
    )
    np.fft.fft(c_coefs)
