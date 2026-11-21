import copy
import unittest
import numpy as np
import solution as impl


class Tests(unittest.TestCase):

    def test_tokenize_and_vocabulary(self):
        self.assertEqual(impl.tokenize('Cat, CAT! Dog 123 tf-idf_2'), ['cat', 'cat', 'dog', 'tf', 'idf'])
        self.assertEqual(impl.build_vocabulary(['b a', 'a c']), {'a': 0, 'b': 1, 'c': 2})
        with self.assertRaises(ValueError):
            impl.build_vocabulary([])
