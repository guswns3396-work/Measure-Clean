import unittest
import numpy as np
import pandas as pd
from process.measure import Measure


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        class TestMeasure(Measure):
            @classmethod
            def get_prefix(cls):
                return 'test'

            @classmethod
            def get_cols(cls):
                return [f'test_{i}' for i in range(5)]

            @classmethod
            def check_range(cls, df):
                pass

            @classmethod
            def score(cls, df):
                pass

        self.TestMeasure = TestMeasure()

    # TODO: implement rest of instantiation


class TestReverseCode(unittest.TestCase):
    # TODO: implement rest of test cases

    def test__standard__reverse_code(self):
        df = pd.DataFrame(
            [[0, 1, 2, 3, 4],
             [4, 3, 2, 1, 0],
             [1, 3, 0, 2, 4]],
            columns=[f'testcol_{i}' for i in range(5)]
        )
        df_target = pd.DataFrame(
            [[4, 3, 2, 1, 0],
             [0, 1, 2, 3, 4],
             [3, 1, 4, 2, 0]],
            columns=[f'testcol_{i}' for i in range(5)]
        )
        df_source = Measure.reverse_code(df, [i for i in range(5)], r'testcol_(\d+)', 0, 4)
        self.assertTrue((df_source == df_target).all().all())

    def test__nan__reverse_code(self):
        pass

# TODO: implement tests for other methods


if __name__ == "__main__":
    unittest.main()
