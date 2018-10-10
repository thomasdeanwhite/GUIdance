import data_loader as ten
import config as cfg
import unittest
import model_test as tester

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

    def test_coordinate_conversion_aspect_one(self):
        aspect = 1.0
        input = [0.1, 0.4, 0.2, 0.5]
        converted = tester.convert_coords(input[0], input[1], input[2], input[3], aspect)
        correct = [0.1, 0.4, 0.2, 0.5]

        for i in range(len(input)):
            self.assertAlmostEqual(correct[i], converted[i])

    def test_coordinate_conversion_aspect_two(self):
        aspect = 2
        input = [0.1, 0.3, 0.2, 0.5]
        converted = tester.convert_coords(input[0], input[1], input[2], input[3], aspect)
        correct = [0.1, 0.1, 0.2, 1.0]

        for i in range(len(input)):
            self.assertAlmostEqual(correct[i], converted[i], msg="expected "+str(correct)+" but got "+str(converted))


    def test_coordinate_conversion_aspect_two_border(self):
        aspect = 2
        input = [0.15, 0.25, 0.3, 0.1]
        converted = tester.convert_coords(input[0], input[1], input[2], input[3], aspect)
        correct = [0.15, 0.0, 0.3, 0.2]

        for i in range(len(input)):
            self.assertAlmostEqual(correct[i], converted[i], msg="expected "+str(correct)+" but got "+str(converted))


    def test_coordinate_conversion_aspect_two(self):
        aspect = 0.5
        input = [0.3, 0.1, 0.2, 0.5]
        converted = tester.convert_coords(input[0], input[1], input[2], input[3], aspect)
        correct = [0.1, 0.1, 0.1, 0.5]

        for i in range(len(input)):
            self.assertAlmostEqual(correct[i], converted[i], msg="expected "+str(correct)+" but got "+str(converted))


    def test_coordinate_conversion_aspect_half_border(self):
        aspect = 0.5
        input = [0.25, 0.25, 0.3, 0.1]
        converted = tester.convert_coords(input[0], input[1], input[2], input[3], aspect)
        correct = [0.0, 0.25, 0.15, 0.1]

        for i in range(len(input)):
            self.assertAlmostEqual(correct[i], converted[i], msg="expected "+str(correct)+" but got "+str(converted))


if __name__ == '__main__':
    unittest.main()