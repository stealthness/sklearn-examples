"""
Test case for boolean masks
"""
import unittest
import random

import numpy as np

from boolmask_experiment import boolean_mask
from boolmask_experiment.boolean_mask import get_mask_from_string

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

    # def test_boolean_operator_and_mask_repeatively(self):
    #     mask_set = self.get_set()
    #     print(mask_set)
    #     # empty_mask = boolean_mask.get_mask(MASK_SIZE, type='empty')
    #     # full_mask = boolean_mask.get_mask(MASK_SIZE, type='full')
    #     # act_mask_0 = boolean_mask.get_mask(4, type='half_0')
    #     # act_mask_1 = boolean_mask.get_mask(4, type='half_1')
    #     b = bool_and(random.choice(mask_set), random.choice(mask_set))
    #     print(b)
    #     for i in range(10):
    #         b = bool_and(b, random.choice(mask_set))
    #         print(f'{i} -- {b}')
    #
    #     print(b)
    #     self.fail()


    @staticmethod
    def get_set():
        empty_mask = boolean_mask.get_mask(MASK_SIZE, type='empty')
        full_mask = boolean_mask.get_mask(MASK_SIZE, type='full')
        act_mask_0 = boolean_mask.get_mask(MASK_SIZE, type='half_0')
        act_mask_1 = boolean_mask.get_mask(MASK_SIZE, type='half_1')
        return [empty_mask, full_mask, act_mask_0, act_mask_1]

    def test_get_mask_from_string_using_1111(self):
        full_mask = boolean_mask.get_mask(4, type='full')
        np.testing.assert_array_equal(full_mask, get_mask_from_string("1111"))

    def test_get_mask_from_string_using_0000(self):
        empty_mask = boolean_mask.get_mask(4, type='empty')
        np.testing.assert_array_equal(empty_mask, get_mask_from_string("0000"))

    def test_get_mask_from_string_using_1100(self):
        half_0_mask = boolean_mask.get_mask(4, type='half_0')
        np.testing.assert_array_equal(half_0_mask, get_mask_from_string("1100"))

    def test_get_mask_from_string_using_0011(self):
        half_1_mask = boolean_mask.get_mask(4, type='half_1')
        np.testing.assert_array_equal(half_1_mask, get_mask_from_string("0011"))