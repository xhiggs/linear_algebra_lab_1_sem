import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):

    def test_fit_transform_manual(self):
        vocabulary, idf, matrix = impl.tfidf_fit(['cat cat sleeps', 'dog sleeps'])
        show("TF-IDF: строки — документы, столбцы — слова словаря",
             documents=['cat cat sleeps', 'dog sleeps'], vocabulary=vocabulary, idf=idf, tfidf=matrix)
        self.assertEqual(vocabulary, {'cat': 0, 'dog': 1, 'sleeps': 2})
        rare = 1 + np.log(1.5)
        np.testing.assert_allclose(matrix, [[2 / 3 * rare, 0, 1 / 3], [0, rare / 2, 0.5]])
        snapshot = idf.copy()
        query = impl.tfidf_transform(['cat unknown'], vocabulary, idf)
        show("Преобразование запроса без переобучения IDF", query='cat unknown', vector=query)
        np.testing.assert_allclose(query, [[rare / 2, 0, 0]])
        np.testing.assert_array_equal(snapshot, idf)
        self.assertEqual(impl.tfidf_transform([], vocabulary, idf).shape, (0, 3))

    def test_empty_vocabulary(self):
        vocab, idf, matrix = impl.tfidf_fit(['', '123'])
        self.assertEqual(vocab, {})
        self.assertEqual(idf.shape, (0,))
        self.assertEqual(matrix.shape, (2, 0))
