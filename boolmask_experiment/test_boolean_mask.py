"""
Test case for boolean masks
"""
import unittest
import numpy as np

from boolmask_experiment import boolean_mask

MASK_SIZE = 16


class TestBooleanMasks(unittest.TestCase):

    def test_true(self):
        self.assertTrue(True)

    def test_boolean_mask_is_all_false(self):
        mask = boolean_mask.get_mask(MASK_SIZE, type='empty')
        self.assertEqual(len(mask), MASK_SIZE)
        for item in mask:
            self.assertFalse(item)

    def test_boolean_mask_is_all_true(self):
        mask = boolean_mask.get_mask(MASK_SIZE, type='full')
        self.assertEqual(len(mask), MASK_SIZE)
        for item in mask:
            self.assertTrue(item)

    def test_boolean_mask_is_all_true(self):
        mask = boolean_mask.get_mask(MASK_SIZE, type='full')
        self.assertEqual(len(mask), MASK_SIZE)
        for item in mask:
            self.assertTrue(item)

    def test_boolean_mask_random(self):
        mask = boolean_mask.get_mask(MASK_SIZE, type='random')
        print(mask)
        self.assertTrue(True)

    def test_operator_and__on_boolean_mask(self):
        empty_mask = boolean_mask.get_mask(MASK_SIZE, type='empty')
        full_mask = boolean_mask.get_mask(MASK_SIZE, type='full')
        np.testing.assert_almost_equal(empty_mask, np.logical_and(empty_mask, full_mask))

    def test_operator_or__on_boolean_mask(self):
        empty_mask = boolean_mask.get_mask(MASK_SIZE, type='empty')
        full_mask = boolean_mask.get_mask(MASK_SIZE, type='full')
        np.testing.assert_almost_equal(full_mask, np.logical_or(empty_mask, full_mask))

    def test_boolean_mask_for_half(self):
        exp_mask = np.array([True, True, False, False])
        act_mask_0 = boolean_mask.get_mask(4, type='half_0')
        act_mask_1 = boolean_mask.get_mask(4, type='half_1')
        self.assertFalse(np.logical_xor(act_mask_0, exp_mask).all())
        self.assertFalse(np.logical_xor(act_mask_1, np.logical_not(exp_mask)).all())
