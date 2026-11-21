import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):

    def test_tf_with_unknown_and_empty(self):
        vocab = {'cat': 0, 'sleeps': 1}
        actual = impl.term_frequency(['cat', 'cat', 'world'], vocab)
        show("TF с неизвестным словом", tokens=['cat', 'cat', 'world'], vocabulary=vocab,
             computed=actual, expected=[2 / 3, 0])
        np.testing.assert_allclose(actual, [2 / 3, 0])
        np.testing.assert_array_equal(impl.term_frequency([], vocab), [0, 0])
