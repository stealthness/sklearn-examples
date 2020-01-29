"""
Test case for boolean masks
"""
import unittest

from boolmask_experiment import boolean_mask

MASK_SIZE = 16


class TestBooleanMasks(unittest.TestCase):

    def test_true(self):
        self.assertTrue(True)

    def test_boolean_mask_is_all_false(self):
        mask = boolean_mask.get_empty_mask(MASK_SIZE)
        self.assertEqual(len(mask), MASK_SIZE)
        for item in mask:
            self.assertFalse(item)

    def test_boolean_mask_is_all_talse(self):
        mask = boolean_mask.get_full_mask(MASK_SIZE)
        self.assertEqual(len(mask), MASK_SIZE)
        for item in mask:
            self.assertTrue(item)
