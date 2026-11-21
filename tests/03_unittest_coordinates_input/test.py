import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):
    def test_coordinates_and_reconstruction(self):
        for basis, vector, expected in [([[1, 1], [1, -1]], [3, 1], [2, 1]),
                                        ([[1, 2], [0, 1]], [5, 2], [1, 2]),
                                        ([[2]], [6], [3])]:
            result = impl.change_coordinates(vector, basis)
            restored = np.array(basis) @ result
            show("Смена координат: P c = x", P=basis, x=vector, c=result, restored=restored)
            self.assertIsInstance(result, list)
            np.testing.assert_allclose(result, expected, atol=1e-9)
            np.testing.assert_allclose(restored, vector, atol=1e-9)

    def test_general_size_and_no_mutation(self):
        for n in [3, 5]:
            basis = (np.eye(n) * 2 + np.triu(np.ones((n, n)), 1)).tolist()
            vector = list(range(1, n + 1))
            saved = copy.deepcopy((basis, vector))
            result = impl.change_coordinates(vector, basis)
            np.testing.assert_allclose(np.array(basis) @ result, vector, atol=1e-9)
            self.assertEqual((basis, vector), saved)

    def test_invalid_basis(self):
        for basis in [[[1, 2], [2, 4]], [[1e-11, 0], [0, 1]]]:
            with self.assertRaises(ValueError):
                impl.change_coordinates([1, 2], basis)
