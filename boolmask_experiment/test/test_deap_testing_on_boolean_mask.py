from unittest import TestCase
import numpy as np

from boolmask_experiment.boolean_mask import get_mask, bool_and, bool_or, get_mask_from_string

TEST_CASE_FILE = 'test_cases_bool_operator.txt'

b0000 = get_mask_from_string("0000")
b1111 = get_mask_from_string("1111")
b1100 = get_mask_from_string("1100")
b0011 = get_mask_from_string("0011")

b1000 = get_mask_from_string("1000")
b0100 = get_mask_from_string("0100")
b0010 = get_mask_from_string("0010")
b0001 = get_mask_from_string("0001")

b0110 = get_mask_from_string("0110")
b0101 = get_mask_from_string("0101")
b1010 = get_mask_from_string("1010")

b1101 = get_mask_from_string("1101")
b1011 = get_mask_from_string("1011")
b0111 = get_mask_from_string("0111")
b1110 = get_mask_from_string("1110")


class Test(TestCase):

    def test_constants(self):
        np.testing.assert_array_equal(np.array([False, False, False, False]), b0000)
        np.testing.assert_array_equal(np.array([True, True, True, True]), b1111)
        np.testing.assert_array_equal(np.array([True, True, False, False]), b1100)
        np.testing.assert_array_equal(np.array([False, False, True, True]), b0011)

    # testing and
    @staticmethod
    def run_testcase(test_case, filename, function):
        with open(filename) as file:
            for line in file:
                if line.startswith(test_case):
                    parts = line.split(',')
                    for i in range(int(parts[1])):
                        bits = file.readline().strip().split(',')
                        exp = get_mask_from_string(bits[0])
                        act = function(get_mask_from_string(bits[1]), get_mask_from_string(bits[2]))
                        np.testing.assert_array_equal(exp, act, f'testcase is {test_case} no {i}')

    def testcase_1_and_with_array_size_1(self):
        self.run_testcase('testcase:1and', TEST_CASE_FILE, bool_and)

    def testcase_21_or_with_array_size_1(self):
        self.run_testcase('testcase:1or', TEST_CASE_FILE, bool_or)



    def test_bool_and_with_half(self):
        np.testing.assert_array_equal(b0000, bool_and(b0000, b0001))
        self.assertFalse(bool_and(b1100, b0011).all())
        self.assertFalse(bool_and(b0011, b1100).all())

    def test_bool_and_with_full_and_empty(self):
        self.assertFalse(bool_and(b1111, b0000).all())
        self.assertFalse(bool_and(b1111, b0000).any())

    def test_bool_and_with_full_and_half(self):
        self.assertFalse(bool_and(b1111, b0111).all())
        self.assertTrue(bool_and(b1111, b0111).any())

        self.assertFalse(bool_and(b1111, b1011).all())
        self.assertTrue(bool_and(b1111, b1011).any())

    def test_bool_or_with_half(self):
        self.assertTrue(bool_or(b1100, b0011).all())
        self.assertTrue(bool_or(b0011, b1100).all())

    # Test or

    def test_bool_or_with_full_or_empty(self):
        self.assertTrue(bool_or(b1111, b0000).all())

    def test_bool_or_with_full_or_half(self):
        self.assertTrue(bool_or(b1111, b0111).all())
        self.assertTrue(bool_or(b1111, b0111).any())

        self.assertTrue(bool_or(b1111, b1011).all())
        self.assertTrue(bool_or(b1111, b1011).any())
