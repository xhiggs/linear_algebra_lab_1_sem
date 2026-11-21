import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):

    def test_determinant_small(self):
        for matrix, expected in [([[7]], 7), ([[1, 2], [3, 4]], -2),
                                 ([[0, 1, 0], [1, 0, 0], [0, 0, 2]], -2)]:
            actual = impl.determinant(matrix)
            show("Определитель", A=matrix, computed=actual, expected=expected)
            self.assertAlmostEqual(actual, expected)

    def test_determinant_arbitrary_size(self):
        rng = np.random.default_rng(19)
        for n in [4, 5, 8, 12]:
            a = rng.integers(-4, 5, (n, n)) + 10 * np.eye(n)
            expected = np.linalg.det(a)
            actual = impl.determinant(a.tolist())
            show(f"Определитель {n}×{n}", computed=actual, expected=expected)
            self.assertTrue(np.isclose(actual, expected, rtol=1e-08))

    def test_singular_and_small_nonzero_pivot(self):
        self.assertAlmostEqual(impl.determinant([[1, 2], [2, 4]]), 0)
        self.assertAlmostEqual(impl.determinant([[1e-12, 0], [0, 1000000000000.0]]), 1)

    def test_determinant_no_mutation(self):
        a = [[0, 2], [3, 4]]
        saved = copy.deepcopy(a)
        impl.determinant(a)
        self.assertEqual(a, saved)
