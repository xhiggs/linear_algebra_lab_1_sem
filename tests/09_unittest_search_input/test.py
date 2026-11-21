import copy
import unittest
import numpy as np
import solution as impl
from tests.diagnostics import show


class Tests(unittest.TestCase):

    def test_ranking(self):
        vocab, idf, matrix = impl.tfidf_fit(['cat sleeps', 'dog runs', 'cat sleeps'])
        result = impl.rank_documents('cat', matrix, vocab, idf, top_k=10)
        show("Поиск: пары (индекс документа, cosine)",
             documents=['cat sleeps', 'dog runs', 'cat sleeps'], query='cat', ranking=result)
        self.assertEqual([i for i, _ in result], [0, 2, 1])
        self.assertEqual(impl.rank_documents('cat', matrix, vocab, idf, top_k=0), [])
        unknown = impl.rank_documents('zzz', matrix, vocab, idf)
        show("Неизвестный запрос", query='zzz', ranking=unknown)
        self.assertEqual(unknown, [(0, 0.0), (1, 0.0), (2, 0.0)])

    def test_empty_vocabulary_ranking(self):
        self.assertEqual(impl.rank_documents('abc', np.zeros((2, 0)), {}, np.zeros(0)), [(0, 0.0), (1, 0.0)])
