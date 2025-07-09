import unittest
import numpy as np
import pandas as pd
from measure_clean.measure import Measure


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


class TestHandleDuplicate(unittest.TestCase):
    def test__standard__first__handle_duplicate(self):
        df = pd.DataFrame([[1, 2, 3, 10], [4, 5, 6, 11]], columns=['a', 'b', 'c', 'c'])
        target = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
        source = Measure.handle_duplicate(df, keep='first')

        self.assertTrue((target == source).all().all())

    def test__standard__last__handle_duplicate(self):
        df = pd.DataFrame([[1, 2, 3, 10], [4, 5, 6, 11]], columns=['a', 'b', 'c', 'c'])
        target = pd.DataFrame([[1, 2, 10], [4, 5, 11]], columns=['a', 'b', 'c'])
        source = Measure.handle_duplicate(df, keep='last')

        self.assertTrue((target == source).all().all())

    def test__no_keep__handle_duplicate(self):
        df = pd.DataFrame([[1, 2, 3, 10], [4, 5, 6, 11]], columns=['a', 'b', 'c', 'c'])
        target = df.T.loc['c']
        try:
            Measure.handle_duplicate(df, keep=None)
        except Exception as e:
            source = e.data

        self.assertTrue((target == source).all().all())

# TODO: implement tests for other methods


if __name__ == "__main__":
    unittest.main()
