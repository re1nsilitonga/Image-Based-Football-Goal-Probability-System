import unittest
import numpy as np
from model import GoalProbabilityCalculator
from utils import FieldProcessor

class TestGoalProbabilitySystem(unittest.TestCase):
    def setUp(self):
        self.gp_calc = GoalProbabilityCalculator()
        self.processor = FieldProcessor()

    def test_base_probability(self):
        prob_close = self.gp_calc.calculate_base_probability(10, 0.5)
        prob_far = self.gp_calc.calculate_base_probability(30, 0.5)
        self.assertGreater(prob_close, prob_far)
        prob_wide = self.gp_calc.calculate_base_probability(15, 1.0)
        prob_narrow = self.gp_calc.calculate_base_probability(15, 0.1)
        self.assertGreater(prob_wide, prob_narrow)

    def test_obstacle_factor_standard(self):
        factor = self.gp_calc._calculate_obstacle_factor(20, 0.5, [])
        self.assertEqual(factor, 1.0)
        defenders = [(5, 0)]
        factor_blocked = self.gp_calc._calculate_obstacle_factor(20, 0.5, defenders)
        self.assertLess(factor_blocked, 1.0)
        defenders_far = [(5, 10)]
        factor_clear = self.gp_calc._calculate_obstacle_factor(20, 0.5, defenders_far)
        self.assertGreater(factor_clear, factor_blocked)

    def test_obstacle_factor_eigenvalue(self):
        wall = [(5, -1), (5, 0), (5, 1)]
        line = [(4, 0), (5, 0), (6, 0)]
        factor_wall = self.gp_calc._calculate_obstacle_factor_with_eigenvalue(20, 0.5, wall)
        factor_line = self.gp_calc._calculate_obstacle_factor_with_eigenvalue(20, 0.5, line)
        self.assertLess(factor_wall, factor_line)

    def test_affine_basis_transform(self):
        src_origin = [0, 0]
        src_a = [10, 0]
        src_b = [0, 10]
        dst_origin = [0, 0]
        dst_a = [10, 0]
        dst_b = [0, 10]
        pts = [[5, 5], [2, 8], [9, 1]]
        mapped = self.processor.transform_points_affine(pts, src_origin, src_a, src_b, dst_origin, dst_a, dst_b)
        for i, p in enumerate(pts):
            np.testing.assert_array_almost_equal(mapped[i], p)

if __name__ == '__main__':
    unittest.main()
