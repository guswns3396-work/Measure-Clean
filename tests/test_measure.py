import unittest
from abc import ABC

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

        self.TestMeasure = TestMeasure

    # TODO: implement rest of instantiation


class TestCheckRange(unittest.TestCase):

    def setUp(self):
        class TestMeasure(Measure):
            @classmethod
            def get_prefix(cls):
                return 'test'

            @classmethod
            def get_cols(cls):
                return [f'test_{i}' for i in range(5)]

            @classmethod
            def get_score_suffixes(cls):
                return ['score']

            @classmethod
            def check_range(cls, df):
                vals = [i for i in range(0, 3 + 1)]
                return cls.argwhere(cls.is_invalid_discrete(df[cls.get_cols()], vals))

            @classmethod
            def score(cls, df):
                score = df.sum(axis=1, skipna=False)
                score.name = f"{cls.get_prefix()}_{cls.get_score_suffixes()[-1]}"
                return score

        self.TestMeasure = TestMeasure
        df = pd.DataFrame(
            [[i for i in range(5)] for j in range(5)],
            columns=TestMeasure.get_cols()
        )
        df = pd.concat([df, TestMeasure.score(df)], axis=1)
        self.df = df

    def test__items_only__check_range(self):
        idx = self.TestMeasure.check_range(self.df[self.TestMeasure.get_cols()])
        self.assertEqual(len(idx), 5)

    def test__scores__check_range(self):
        idx = self.TestMeasure.check_range(self.df)
        self.assertEqual(len(idx), 5)


class TestReverseCode(unittest.TestCase):
    # TODO: implement rest of test cases

    def test__standard__reverse_code(self):
        df = pd.DataFrame(
            [[0, 1, 2, 3, 4],
             [4, 3, 2, 1, 0],
             [1, 3, 0, 2, 4]],
            columns=[f'testcol_{i}' for i in range(5)]
        )
        df['ID'] = 0
        df['SES'] = 1
        df['AGE'] = 2
        df = df.set_index(['ID', 'SES', 'AGE'])
        df_target = pd.DataFrame(
            [[4, 3, 2, 1, 0],
             [0, 1, 2, 3, 4],
             [3, 1, 4, 2, 0]],
            columns=[f'testcol_{i}' for i in range(5)]
        )
        df_target['ID'] = 0
        df_target['SES'] = 1
        df_target['AGE'] = 2
        df_target = df_target.set_index(['ID', 'SES', 'AGE'])
        df_source = Measure.reverse_code(df, [i for i in range(5)], r'testcol_(\d+)', 0, 4)
        self.assertTrue((df_source == df_target).all().all())

    def test__nan__reverse_code(self):
        # TODO
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
        target = df['c']
        try:
            Measure.handle_duplicate(df, keep=None)
        except Exception as e:
            source = e.data
        self.assertTrue((target == source).all().all())

    def test__last_all_nans__handle_duplicate(self):
        df = pd.DataFrame(
            [[1, 2, 3, 10, np.nan, 11],
             [4, 5, 6, 11, np.nan, 12]],
            columns=['a', 'b', 'c', 'd', 'c', 'd']
        )
        target = pd.DataFrame(
            [[1, 2, np.nan, 11],
             [4, 5, np.nan, 12]],
            columns=['a', 'b', 'c', 'd']
        )
        source = Measure.handle_duplicate(df, keep='last')
        self.assertTrue(source.astype(float).equals(target.astype(float)))

    def test__first_all_nans__handle_duplicate(self):
        df = pd.DataFrame(
            [[1, 2, np.nan, 10, 3, 11],
             [4, 5, np.nan, 11, 4, 12]],
            columns=['a', 'b', 'c', 'd', 'c', 'd']
        )
        target = pd.DataFrame(
            [[1, 2, np.nan, 10],
             [4, 5, np.nan, 11]],
            columns=['a', 'b', 'c', 'd']
        )
        source = Measure.handle_duplicate(df, keep='first')
        self.assertTrue(source.astype(float).equals(target.astype(float)))


class TestCalculateAge(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame([['MDMA001', 'events_and_logs_arm_2', np.nan, np.nan, '2000-1-1', '2001-1-1', '2002-1-1',
                            np.nan, np.nan, np.nan],
                           ['MDMA001', 's1_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-1-1', 1.0, 1.0],
                           ['MDMA001', 's2_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-2-1', 1.0, 1.0],
                           ['MDMA001', 's3_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-3-1', 1.0, 1.0],
                           ['MDMA002', 'events_and_logs_arm_2', np.nan, np.nan, '2000-2-1', '2001-2-1', '2002-2-1',
                            np.nan, np.nan, np.nan],
                           ['MDMA001', 'screening_visit_arm_2', np.nan, '1900-1-1', np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan],
                           ['MDMA002', 's1_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-4-1', 3.0, 2.0],
                           ['MDMA002', 's2_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-5-1', 2.0, 1.0],
                           ['MDMA002', 's3_preadmin_presca_arm_2', np.nan, np.nan, np.nan, np.nan, np.nan,
                            '2010-6-1', 1.0, 1.0],
                           ['MDMA002', 'screening_visit_arm_2', 30.0, '1910-1-1', np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan]])
        df.columns = ['ID', 'SES', 'AGE', 'dob', 'shced_date_baseline',
                      'shced_date_baseline2', 'shced_date_baseline3', 'sesscrn_dateobtained',
                      'brisc1', 'brisc2']
        self.df = df.set_index(['ID', 'SES', 'AGE'])

    def test__different_row__replace__calculate_age(self):
        mapping = {
            's1_preadmin_presca_arm_2': 'shced_date_baseline',
            's2_preadmin_presca_arm_2': 'shced_date_baseline2',
            's3_preadmin_presca_arm_2': 'shced_date_baseline3'
        }
        source = Measure.calculate_age(
            self.df, how='replace', date_col=mapping, dob_col='dob'
        ).index.values
        source = pd.DataFrame([list(t) for t in source]).set_index([0, 1])
        target = pd.DataFrame([
            ['MDMA001', 'events_and_logs_arm_2', np.nan],
            ['MDMA001', 's1_preadmin_presca_arm_2', 99],
            ['MDMA001', 's2_preadmin_presca_arm_2', 100],
            ['MDMA001', 's3_preadmin_presca_arm_2', 101],
            ['MDMA001', 'screening_visit_arm_2', np.nan],
            ['MDMA002', 'events_and_logs_arm_2', np.nan],
            ['MDMA002', 's1_preadmin_presca_arm_2', 90],
            ['MDMA002', 's2_preadmin_presca_arm_2', 91],
            ['MDMA002', 's3_preadmin_presca_arm_2', 92],
            ['MDMA002', 'screening_visit_arm_2', np.nan],
        ]).set_index([0, 1])
        self.assertTrue(source.sort_index().equals(target.sort_index()))

    def test__different_row__fill__calculate_age(self):
        mapping = {
            's1_preadmin_presca_arm_2': 'shced_date_baseline',
            's2_preadmin_presca_arm_2': 'shced_date_baseline2',
            's3_preadmin_presca_arm_2': 'shced_date_baseline3'
        }
        source = Measure.calculate_age(
            self.df, how='fill', date_col=mapping, dob_col='dob'
        ).index.values
        source = pd.DataFrame([list(t) for t in source]).set_index([0, 1])
        target = pd.DataFrame([
            ['MDMA001', 'events_and_logs_arm_2', np.nan],
            ['MDMA001', 's1_preadmin_presca_arm_2', 99],
            ['MDMA001', 's2_preadmin_presca_arm_2', 100],
            ['MDMA001', 's3_preadmin_presca_arm_2', 101],
            ['MDMA001', 'screening_visit_arm_2', np.nan],
            ['MDMA002', 'events_and_logs_arm_2', np.nan],
            ['MDMA002', 's1_preadmin_presca_arm_2', 90],
            ['MDMA002', 's2_preadmin_presca_arm_2', 91],
            ['MDMA002', 's3_preadmin_presca_arm_2', 92],
            ['MDMA002', 'screening_visit_arm_2', 30],
        ]).set_index([0, 1])
        self.assertTrue(source.sort_index().equals(target.sort_index()))

    def test__same_row__replace__calculate_age(self):
        source = Measure.calculate_age(
            self.df, how='replace', date_col='sesscrn_dateobtained', dob_col='dob'
        ).index.values
        source = pd.DataFrame([list(t) for t in source]).set_index([0, 1])
        target = pd.DataFrame([
            ['MDMA001', 'events_and_logs_arm_2', np.nan],
            ['MDMA001', 's1_preadmin_presca_arm_2', 109],
            ['MDMA001', 's2_preadmin_presca_arm_2', 110],
            ['MDMA001', 's3_preadmin_presca_arm_2', 110],
            ['MDMA001', 'screening_visit_arm_2', np.nan],
            ['MDMA002', 'events_and_logs_arm_2', np.nan],
            ['MDMA002', 's1_preadmin_presca_arm_2', 100],
            ['MDMA002', 's2_preadmin_presca_arm_2', 100],
            ['MDMA002', 's3_preadmin_presca_arm_2', 100],
            ['MDMA002', 'screening_visit_arm_2', np.nan],
        ]).set_index([0, 1])
        self.assertTrue(source.sort_index().equals(target.sort_index()))

    def test__same_row__fill__calculate_age(self):
        source = Measure.calculate_age(
            self.df, how='fill', date_col='sesscrn_dateobtained', dob_col='dob'
        ).index.values
        source = pd.DataFrame([list(t) for t in source]).set_index([0, 1])
        target = pd.DataFrame([
            ['MDMA001', 'events_and_logs_arm_2', np.nan],
            ['MDMA001', 's1_preadmin_presca_arm_2', 109],
            ['MDMA001', 's2_preadmin_presca_arm_2', 110],
            ['MDMA001', 's3_preadmin_presca_arm_2', 110],
            ['MDMA001', 'screening_visit_arm_2', np.nan],
            ['MDMA002', 'events_and_logs_arm_2', np.nan],
            ['MDMA002', 's1_preadmin_presca_arm_2', 100],
            ['MDMA002', 's2_preadmin_presca_arm_2', 100],
            ['MDMA002', 's3_preadmin_presca_arm_2', 100],
            ['MDMA002', 'screening_visit_arm_2', 30],
        ]).set_index([0, 1])
        self.assertTrue(source.sort_index().equals(target.sort_index()))


class TestScoreIfNeeded(unittest.TestCase):

    def setUp(self):
        class TestMeasure(Measure):
            @classmethod
            def get_prefix(cls):
                return 'test'

            @classmethod
            def get_cols(cls):
                return [f'test_{i}' for i in range(5)]

            @classmethod
            def get_score_suffixes(cls):
                return ['score']

            @classmethod
            def check_range(cls, df):
                vals = [i for i in range(0, 3 + 1)]
                return cls.argwhere(cls.is_invalid_discrete(df[cls.get_cols()], vals))

            @classmethod
            def score(cls, df):
                score = df.sum(axis=1, skipna=False)
                score.name = f"{cls.get_prefix()}_{cls.get_score_suffixes()[-1]}"
                return score

        self.TestMeasure = TestMeasure
        df = pd.DataFrame(
            [[i for i in range(5)] for j in range(5)],
            columns=TestMeasure.get_cols()
        )
        df = pd.concat([df, TestMeasure.score(df)], axis=1)
        self.df = df

    def test__items_only__score_if_needed(self):
        df = self.TestMeasure.score_if_needed(self.df[self.TestMeasure.get_cols()], keep=None)
        self.assertTrue(len(df.columns[df.columns == 'test_score']) == 1)

    def test__scores__score_if_needed(self):
        try:
            df = self.TestMeasure.score_if_needed(self.df, keep=None)
        except Exception as e:
            source = e.data
        target = pd.concat([self.df['test_score'], self.TestMeasure.score(self.df)], axis=1)
        self.assertTrue((source == target).all().all())

# TODO: implement tests for other methods


if __name__ == "__main__":
    unittest.main()
