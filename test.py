import unittest
import numpy as np
from model import XGCalculator
from utils import FieldProcessor

class TestXGSystem(unittest.TestCase):
    def setUp(self):
        self.xg_calc = XGCalculator()
        self.processor = FieldProcessor()

    def test_base_xg(self):
        xg_close = self.xg_calc.calculate_base_xg(10, 0.5)
        xg_far = self.xg_calc.calculate_base_xg(30, 0.5)
        self.assertGreater(xg_close, xg_far)
        xg_wide = self.xg_calc.calculate_base_xg(15, 1.0)
        xg_narrow = self.xg_calc.calculate_base_xg(15, 0.1)
        self.assertGreater(xg_wide, xg_narrow)

    def test_obstacle_factor_standard(self):
        factor = self.xg_calc._calculate_obstacle_factor(20, 0.5, [])
        self.assertEqual(factor, 1.0)
        defenders = [(5, 0)]
        factor_blocked = self.xg_calc._calculate_obstacle_factor(20, 0.5, defenders)
        self.assertLess(factor_blocked, 1.0)
        defenders_far = [(5, 10)]
        factor_clear = self.xg_calc._calculate_obstacle_factor(20, 0.5, defenders_far)
        self.assertGreater(factor_clear, factor_blocked)

    def test_obstacle_factor_eigenvalue(self):
        wall = [(5, -1), (5, 0), (5, 1)]
        line = [(4, 0), (5, 0), (6, 0)]
        factor_wall = self.xg_calc._calculate_obstacle_factor_with_eigenvalue(20, 0.5, wall)
        factor_line = self.xg_calc._calculate_obstacle_factor_with_eigenvalue(20, 0.5, line)
        self.assertLess(factor_wall, factor_line)

    def test_homography_transform(self):
        src = [[0,0], [10,0], [0,10], [10,10]]
        dst = [[0,0], [10,0], [0,10], [10,10]]
        H = self.processor.compute_homography(src, dst)
        pt = self.processor.transform_points([(5,5)], H)[0]
        np.testing.assert_array_almost_equal(pt, [5,5])

if __name__ == '__main__':
    unittest.main()
