import numpy as np
import unittest
from model import XGCalculator

class TestXGCalculator(unittest.TestCase):

    def setUp(self):
        # Inisialisasi objek XGCalculator sebelum setiap tes
        self.calculator = XGCalculator()

    def test_calculate_base_xg(self):
        # Input: jarak = 11 meter (penalti), sudut = 0 radian (lurus ke gawang)
        distance = 11
        angle = 0
        expected_xg = 1 / (1 + np.exp(-(2.0 + (-0.15 * distance) + (1.2 * angle))))
        result = self.calculator.calculate_base_xg(distance, angle)
        self.assertAlmostEqual(result, expected_xg, places=5, msg="Base xG tidak sesuai untuk penalti")

    def test_calculate_final_xg_no_defenders(self):
        # Input: jarak = 16 meter, sudut = 0.5 radian, tanpa penghalang
        distance = 16
        angle = 0.5
        defenders = []
        result = self.calculator.calculate_final_xg(distance, angle, defenders)
        self.assertAlmostEqual(result['final_xg'], result['base_xg'], places=5, msg="xG akhir tanpa penghalang tidak sesuai")

    def test_calculate_final_xg_with_defenders(self):
        # Input: jarak = 20 meter, sudut = 0.3 radian, dengan penghalang di posisi tertentu
        distance = 20
        angle = 0.3
        defenders = [(10, 1), (15, -0.5)] # Penghalang di jarak 10m dan 15m dengan offset lateral
        result = self.calculator.calculate_final_xg(distance, angle, defenders)
        self.assertLess(result['final_xg'], result['base_xg'], msg="xG akhir dengan penghalang tidak sesuai")

if __name__ == '__main__':
    unittest.main()