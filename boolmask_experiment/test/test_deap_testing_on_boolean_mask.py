from unittest import TestCase
import numpy as np

from boolmask_experiment.boolean_mask import get_mask, bool_and, bool_or, get_mask_from_string, bool_xor, bool_not

TEST_CASE_FILE = 'test_cases_bool_operator.txt'


class Test(TestCase):

    # testing and
    @staticmethod
    def run_test_case(test_case, filename, function):
        with open(filename) as file:
            for line in file:
                if line.startswith(test_case):
                    parts = line.split(',')
                    for i in range(int(parts[1])):
                        bits = file.readline().strip().split(',')
                        exp = get_mask_from_string(bits[0])
                        print(f'len {len(bits)}  and bits is {bits}')
                        if len(bits) == 2:
                            act = function(get_mask_from_string(bits[1]))
                        else:
                            act = function(get_mask_from_string(bits[1]), get_mask_from_string(bits[2]))
                        np.testing.assert_array_equal(exp, act, f'test-case is {test_case} no {i}')

    def test_case_1_and_with_array_size_1(self):
        self.run_test_case('test-case:1and', TEST_CASE_FILE, bool_and)

    def test_case_1_or_with_array_size_1(self):
        self.run_test_case('test-case:1or', TEST_CASE_FILE, bool_or)

    def test_case_1_xr_with_array_size_1(self):
        self.run_test_case('test-case:1xor', TEST_CASE_FILE, bool_xor)

    def test_case_1_not_with_array_size_1(self):
        self.run_test_case('test-case:1not', TEST_CASE_FILE, bool_not)

    def test_case_2_not_with_array_size_1(self):
        self.run_test_case('testcase:2not', TEST_CASE_FILE, bool_not)

    def test_case_2_and_with_array_size_2(self):
        self.run_test_case('testcase:2and', TEST_CASE_FILE, bool_and)

    def test_case_2_or_with_array_size_2(self):
        self.run_test_case('testcase:2or', TEST_CASE_FILE, bool_or)

    def test_case_2_xor_with_array_size_2(self):
        self.run_test_case('testcase:2xor', TEST_CASE_FILE, bool_xor)

    def test_case_3_and_with_array_size_3(self):
        self.run_test_case('testcase:3and', TEST_CASE_FILE, bool_and)
