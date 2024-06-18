import unittest
import logging
import pytest
import numpy as np
from stellacode.surface.utils import *
from stellacode.tools.vmec import *


class TestCoords(unittest.TestCase):
    def setUp(self):
        pass

    # Convert zero x and zero y coordinates to polar coordinates
    def test_zero_coordinates(self):
        x, y = 0, 0
        r, phi = to_polar(x, y)
        assert r == 0
        assert phi == 0

    # Convert positive x and y coordinates to polar coordinates
    def test_positive_coordinates(self):
        x, y = 3, 4
        r, phi = to_polar(x, y)
        assert r == pytest.approx(5)
        assert phi == pytest.approx(np.arctan2(4, 3))

    # converts polar coordinates with positive radius and angle to Cartesian coordinates correctly
    def test_positive_radius_and_angle(self):
        r = 5.0
        theta = np.pi / 4  # 45 degrees
        x, y = from_polar(r, theta)
        assert np.isclose(x, 3.5355339059327378)
        assert np.isclose(y, 3.5355339059327378)

    # handles negative radius and any angle, returning the correct Cartesian coordinates
    def test_negative_radius(self):
        r = -5.0
        theta = np.pi / 4  # 45 degrees
        x, y = from_polar(r, theta)
        assert np.isclose(x, -3.5355339059327378)
        assert np.isclose(y, -3.5355339059327378)

    # rotating a point by 0 radians returns the same point
    def test_rotate_zero_radians(self):
        x, y = 1.0, 1.0
        angle = 0
        rot_x, rot_y = rotate(x, y, angle)
        assert rot_x == x
        assert rot_y == y

    # rotating the origin (0, 0) by any angle returns (0, 0)
    def test_rotate_origin_any_angle(self):
        x, y = 0.0, 0.0
        angle = 1.57  # Any angle in radians
        rot_x, rot_y = rotate(x, y, angle)
        assert rot_x == x
        assert rot_y == y

    # converts positive Cartesian coordinates correctly
    def test_converts_positive_cartesian_coordinates_correctly(self):
        # Define positive Cartesian coordinates
        xyz = np.array([3.0, 4.0, 5.0])

        # Expected cylindrical coordinates
        expected = np.array([5.0, np.arctan2(4.0, 3.0), 5.0])

        # Convert to cylindrical coordinates
        result = cartesian_to_cylindrical(xyz)

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(result, expected)

    # handles very large values without overflow
    def test_handles_very_large_values_without_overflow(self):
        # Define very large Cartesian coordinates
        large_value = 1e308
        xyz = np.array([large_value, large_value, large_value])

        # Convert to cylindrical coordinates
        result = cartesian_to_cylindrical(xyz)

        # Check that the radial distance is correct and no overflow occurs
        expected_r = np.sqrt(large_value**2 + large_value**2)
        expected_phi = np.arctan2(large_value, large_value)

        assert np.isfinite(result[0]) and result[0] == expected_r
        assert np.isfinite(result[1]) and result[1] == expected_phi
        assert np.isfinite(result[2]) and result[2] == large_value

    # Convert standard 3D Cartesian coordinates to toroidal coordinates
    def test_convert_standard_cartesian_to_toroidal(self):
        xyz = np.array([1.0, 1.0, 1.0])
        tore_radius = 1.0
        tore_height = 1.0
        result = cartesian_to_toroidal(xyz, tore_radius, tore_height)
        expected_shape = (3,)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"

    # Handle zero vector input (0, 0, 0)
    def test_handle_zero_vector_input(self):
        xyz = np.array([0.0, 0.0, 0.0])
        tore_radius = 1.0
        tore_height = 1.0
        result = cartesian_to_toroidal(xyz, tore_radius, tore_height)
        expected_result = np.array([1.0, 0.0, 0.0])
        assert np.allclose(
            result, expected_result), f"Expected {expected_result}, got {result}"

    # Convert simple Cartesian coordinates with no shift or rotation
    def test_convert_simple_cartesian_no_shift_or_rotation(self):
        xyz = np.array([1.0, 1.0, 1.0])
        expected = np.array([np.sqrt(2), np.pi / 4, 1.0])
        result = cartesian_to_shifted_cylindrical(xyz)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Convert Cartesian coordinates at the origin (0,0,0)
    def test_convert_cartesian_origin(self):
        xyz = np.array([0.0, 0.0, 0.0])
        expected = np.array([0.0, 0.0, 0.0])
        result = cartesian_to_shifted_cylindrical(xyz)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Correct length calculation for simple constant radius function
    def test_correct_length_constant_radius(self):
        # Define a simple constant radius function r = 2 and its derivative dr/dphi = 0
        def radius_function(phi): return 2
        def dr_dphi_function(phi): return 0

        # Calculate the length of the curve from 0 to pi
        phi = np.pi
        expected_length = 2 * phi  # Analytical solution for this case

        # Invoke the function
        calculated_length = unwrap_u(radius_function, dr_dphi_function, phi)

        # Assert the calculated length is close to the expected length
        assert np.isclose(calculated_length, expected_length, atol=1e-5)
