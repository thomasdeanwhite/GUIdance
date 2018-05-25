import config
import tensor as ten
import unittest

class TestTensor(unittest.TestCase):

    def test_centre(self):
        self.assertAlmostEqual(0, ten.normalise_point(0.5, 3))

    def test_above(self):
        self.assertAlmostEqual(0.1, ten.normalise_point(0.6, 3))

    def test_boundary(self):
        self.assertAlmostEqual(0.16666666, ten.normalise_point(0.66666666, 3))

    def test_below(self):
        self.assertAlmostEqual(-0.1, ten.normalise_point(0.4, 3))

if __name__ == '__main__':
    unittest.main()