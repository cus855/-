"""Independent checks of the equations and integration, using unittest."""

import unittest

import numpy as np

from core import eq12, eq15_rhs, integrate_eq15


class EquationChecks(unittest.TestCase):
    def test_three_layer_ode_matches_closed_form(self):
        times = np.linspace(0, 15, 15001)
        exact = eq12(times, 1.0, 0.001)
        numerical = integrate_eq15(times, 1.0, 0.001, 3)
        self.assertLess(float(np.max(np.abs(exact - numerical))), 1e-8)

    def test_equilibrium_and_positive_growth(self):
        for depth in (3, 5, 100):
            self.assertEqual(eq15_rhs(1.0, 1.0, depth), 0.0)
            self.assertGreater(eq15_rhs(0.01, 1.0, depth), 0.0)
            self.assertLess(eq15_rhs(1.1, 1.0, depth), 0.0)


if __name__ == "__main__":
    unittest.main()
