import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):

    def test_cramer_and_residual(self):
        rng = np.random.default_rng(9)
        for n in [1, 2, 3, 5, 7]:
            a = rng.integers(-3, 4, (n, n)) + 8 * np.eye(n)
            x = rng.normal(size=n)
            rhs = a @ x
            show(f"СЛАУ {n}×{n}: A x = b", A=a, b=rhs)
            actual = impl.solve_cramer(a.tolist(), rhs.tolist())
            show("Решение проверяемой функции", x_student=actual, x_expected=x,
                 residual=a @ actual - rhs, max_residual=float(np.max(np.abs(a @ actual - rhs))))
            np.testing.assert_allclose(actual, x, atol=1e-08)
            np.testing.assert_allclose(a @ actual, rhs, atol=1e-08)
        np.testing.assert_allclose(impl.solve_cramer([[0, 1], [2, 3]], [2, 8]), [1, 2])

    def test_cramer_singular(self):
        for rhs in [[3, 6], [3, 7]]:
            show("Вырожденная система: ожидается ValueError", A=[[1, 2], [2, 4]], b=rhs)
            with self.assertRaises(ValueError):
                impl.solve_cramer([[1, 2], [2, 4]], rhs)
            print("    Получен ожидаемый ValueError.", flush=True)
        with self.assertRaises(ValueError):
            impl.solve_cramer([[1e-11]], [1])

    def test_no_mutation(self):
        a, b = ([[0, 2], [3, 4]], [1, 5])
        saved = copy.deepcopy((a, b))
        impl.determinant(a)
        impl.solve_cramer(a, b)
        self.assertEqual((a, b), saved)
