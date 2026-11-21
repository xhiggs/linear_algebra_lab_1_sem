import copy
import unittest
import numpy as np
import solution as impl


class Tests(unittest.TestCase):

    def test_idf_counts_documents(self):
        actual = impl.inverse_document_frequency(['a a a b', 'b c'], {'a': 0, 'b': 1, 'c': 2})
        np.testing.assert_allclose(actual, [1 + np.log(1.5), 1, 1 + np.log(1.5)])
