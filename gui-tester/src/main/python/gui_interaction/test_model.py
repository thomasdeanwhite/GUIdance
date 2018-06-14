import model_train as ten
import config as cfg
import unittest

class TestTensor(unittest.TestCase):

    # test creation of input from files ----------
    def test_point_centre(self):
        self.assertAlmostEqual(1.5, ten.normalise_point(0.5, 3)[0])

    def test_point_above(self):
        self.assertAlmostEqual(1.8, ten.normalise_point(0.6, 3)[0])

    def test_point_boundary(self):
        self.assertAlmostEqual(2, ten.normalise_point(0.66666666, 3)[0])

    def test_label_normalise(self):
        cfg.grid_shape = [10, 10]
        label = [0.1, 0.4, 0.3, 0.6, 1.0]
        label_norm, centres = ten.normalise_label(label)

        true_norm = [1, 4, 3, 6, 1]

        for i in range(len(label_norm)):
            self.assertAlmostEqual(true_norm[i], label_norm[i])
        # as using factor of 10 for grid_shape, the centres are
        # equal to the position values!
        self.assertAlmostEqual(true_norm[0], centres[0])
        self.assertAlmostEqual(true_norm[1], centres[1])


if __name__ == '__main__':
    unittest.main()