import copy
import unittest
import numpy as np
import solution as impl


class Tests(unittest.TestCase):

    def test_cosine(self):
        self.assertAlmostEqual(impl.cosine_similarity(np.array([3.0, 4]), np.array([6.0, 8])), 1)
        self.assertAlmostEqual(impl.cosine_similarity(np.array([1.0, 0]), np.array([-1.0, 0])), -1)
        self.assertEqual(impl.cosine_similarity(np.zeros(2), np.ones(2)), 0)
