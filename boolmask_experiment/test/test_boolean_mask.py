"""
Test case for boolean masks
"""
import unittest
import random
import numpy as np
from boolmask_experiment.boolean_mask import mask_to_string, get_mask_from_string, get_mask, bool_and, bool_or, \
    bool_xor, bool_not

LOGGING = False
MASK_SIZE = 16

class TestBooleanMasks(unittest.TestCase):

    def test_boolean_mask_is_all_false(self):
        self.assertFalse(get_mask(MASK_SIZE, type='empty').any())

    def test_boolean_mask_is_all_true(self):
        self.assertTrue(get_mask(MASK_SIZE, type='full').all())

    def test_boolean_mask_random(self):
        mask = get_mask(MASK_SIZE, type='random')
        print(f'A random mask is {mask}\n')
        self.assertTrue(True)

    def test_operator_and__on_boolean_mask(self):
        empty_mask = get_mask(MASK_SIZE, type='empty')
        full_mask = get_mask(MASK_SIZE, type='full')
        np.testing.assert_array_equal(empty_mask, np.logical_and(empty_mask, full_mask))

    def test_operator_or__on_boolean_mask(self):
        empty_mask = get_mask(MASK_SIZE, type='empty')
        full_mask = get_mask(MASK_SIZE, type='full')
        np.testing.assert_array_equal(full_mask, np.logical_or(empty_mask, full_mask))

    def test_boolean_mask_for_half(self):
        exp_mask = np.array([True, True, False, False])
        act_mask_0 = get_mask(4, type='half_0')
        act_mask_1 = get_mask(4, type='half_1')
        self.assertFalse(np.logical_xor(act_mask_0, exp_mask).all())
        self.assertFalse(np.logical_xor(act_mask_1, np.logical_not(exp_mask)).all())

    def test_boolean_operator_and_mask_repetitively (self):
        mask_set = self.get_set()
        b = bool_and(get_mask(MASK_SIZE, type='random'), get_mask(MASK_SIZE, type='random'))
        for i in range(100):
            b = bool_and(b, get_mask(MASK_SIZE, type='random'))
            if LOGGING:
                print(f'After multiple operation we have {b}')
        self.assertTrue(True)

    def test_random_boolean_operator_and_mask_repetitively (self):
        function_set = self.get_functions()
        function = random.choice(function_set)
        b = function(random.choice(get_mask(MASK_SIZE, type='random')), get_mask(MASK_SIZE, type='random'))
        for i in range(100):
            print(b)
            function = random.choice(function_set)
            if function == bool_not:
                b = function(b)
            else:
                b = function(b, get_mask(MASK_SIZE, type='random'))
        if LOGGING:
            print(f'After multiple operation we have {b}')
        self.assertTrue(True)


    @staticmethod
    def get_set():
        empty_mask = get_mask(MASK_SIZE, type='empty')
        full_mask = get_mask(MASK_SIZE, type='full')
        act_mask_0 = get_mask(MASK_SIZE, type='half_0')
        act_mask_1 = get_mask(MASK_SIZE, type='half_1')
        return [empty_mask, full_mask, act_mask_0, act_mask_1]

    @staticmethod
    def get_functions():
        return [bool_and, bool_or, bool_xor, bool_not]

    def test_get_mask_from_string_using_1111(self):
        full_mask = get_mask(4, type='full')
        np.testing.assert_array_equal(full_mask, get_mask_from_string("1111"))

    def test_get_mask_from_string_using_0000(self):
        empty_mask = get_mask(4, type='empty')
        np.testing.assert_array_equal(empty_mask, get_mask_from_string("0000"))

    def test_get_mask_from_string_using_1100(self):
        half_0_mask = get_mask(4, type='half_0')
        np.testing.assert_array_equal(half_0_mask, get_mask_from_string("1100"))

    def test_get_mask_from_string_using_0011(self):
        half_1_mask = get_mask(4, type='half_1')
        np.testing.assert_array_equal(half_1_mask, get_mask_from_string("0011"))

    def test_mask_to_string_from_1111(self):
        full_mask = get_mask(4, type='full')
        self.assertEqual("1111", mask_to_string(full_mask))

    def test_mask_to_string_from_0000(self):
        empty_mask = get_mask(4, type='empty')
        self.assertEqual("0000", mask_to_string(empty_mask))

    def test_mask_to_string_from_1100(self):
        half_0_mask = get_mask(4, type='half_0')
        self.assertEqual("1100", mask_to_string(half_0_mask))

    def test_mask_to_string_from_0011(self):
        half_1_mask = get_mask(4, type='half_1')
        self.assertEqual("0011", mask_to_string(half_1_mask))