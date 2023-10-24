import numpy as np

from stellacode.tools.concat_dict import ScaleDictArray


def test_scale_dict():
    sda = ScaleDictArray()

    _dict = {"a": 1e9, "b": 1e-5, "c": np.array([1, 2, 3])}
    s_dict = sda.apply(_dict)

    assert abs(np.linalg.norm(s_dict["c"] - np.mean(s_dict["c"])) - sda.additional_scale) < 1e-15
    assert abs(s_dict["a"] - sda.additional_scale) < 1e-15
    assert abs(s_dict["b"] - sda.additional_scale) < 1e-15

    new_dict = sda.unapply(s_dict)
    for k in s_dict.keys():
        np.testing.assert_almost_equal(_dict[k], new_dict[k])
