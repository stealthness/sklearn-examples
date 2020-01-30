from unittest import TestCase
import numpy as np

from boolmask_experiment.boolean_mask import get_mask, bool_and, bool_or


empty_mask = get_mask(str="0000")
full_mask = get_mask(str="1111")
half_0_mask = get_mask(str="1100")
half_1_mask = get_mask(str="0011")

b1000 = get_mask(str="0011")
b0100 = get_mask(str="0011")
b0010 = get_mask(str="0011")
b0001 = get_mask(str="0011")

b0110 = get_mask(str="0110")
b0101 = get_mask(str="0011")
b1010 = get_mask(str="0011")

b1101 = get_mask(str="1101")
b1011 = get_mask(str="1011")

class Test(TestCase):

    def test_constants(self):
        np.testing.assert_array_equal(np.array([False, False, False, False]), empty_mask)
        np.testing.assert_array_equal(np.array([True, True, True, True]), full_mask)
        np.testing.assert_array_equal(np.array([True, True, False, False]), half_0_mask)
        np.testing.assert_array_equal(np.array([False, False, True, True]), half_1_mask)

    # testing and

    def test_bool_and_with_half(self):
        self.assertFalse(bool_and(half_0_mask, half_1_mask).all())
        self.assertFalse(bool_and(half_1_mask, half_0_mask).all())

    def test_bool_and_with_full_and_empty(self):
        self.assertFalse(bool_and(full_mask, empty_mask).all())
        self.assertFalse(bool_and(full_mask, empty_mask).any())

    def test_bool_and_with_full_and_half(self):
        self.assertFalse(bool_and(full_mask, [False, True, True, True]).all())
        self.assertTrue(bool_and(full_mask, [False, True, True, True]).any())

        self.assertFalse(bool_and(full_mask, [True, False, True, True]).all())
        self.assertTrue(bool_and(full_mask, [True, False, True, True]).any())

    def test_bool_or_with_half(self):
        self.assertTrue(bool_or(half_0_mask, half_1_mask).all())
        self.assertTrue(bool_or(half_1_mask, half_0_mask).all())

    # Test or

    def test_bool_or_with_full_or_empty(self):
        self.assertTrue(bool_or(full_mask, empty_mask).all())

    def test_bool_or_with_full_or_half(self):
        self.assertTrue(bool_or(full_mask, [False, True, True, True]).all())
        self.assertTrue(bool_or(full_mask, [False, True, True, True]).any())

        self.assertTrue(bool_or(full_mask, [True, False, True, True]).all())
        self.assertTrue(bool_or(full_mask, [True, False, True, True]).any())


