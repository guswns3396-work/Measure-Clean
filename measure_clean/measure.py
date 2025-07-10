from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ExceptionWithData(Exception):
    def __init__(self, message, data):
        super().__init__(message)
        self.data = data


class Measure(ABC):
    @classmethod
    @abstractmethod
    def get_prefix(cls):
        pass

    @classmethod
    @abstractmethod
    def get_score_suffixes(cls):
        """
        :return: list of suffixes for score variable name
        """
        pass

    @classmethod
    @abstractmethod
    def get_cols(cls):
        pass

    @classmethod
    @abstractmethod
    def check_range(cls, df):
        """
        checks range and returns indices of invalid values
        :param df: DataFrame of data
        :return: pd.DataFrame where each row indicates indices where df is True
        """
        pass

    @classmethod
    @abstractmethod
    def score(cls, df):
        """
        scores the measure
        :param df: DataFrame of data (should match the expected coding of scoring guide)
        :return: pd.Series or pd.DataFrame of scores
        """
        pass

    @staticmethod
    def get_index():
        return pd.Index(['ID', 'SES', 'AGE'])

    @staticmethod
    def reverse_code(df, col_num, re_str, col_min, col_max):
        """
        reverse codes the data
        :param df: DataFrame of data
        :param col_num: list of column numbers you want to reverse code
        :param re_str: regex to extract column numbers
        :param col_min: variable minimum
        :param col_max: variable maximum
        :return: DataFrame of reverse coded data
        """
        cols = df.columns[df.columns.str.extract(re_str)[0].astype(float).isin(col_num)]
        df.loc[:, cols] = df.loc[:, cols].apply(lambda s: col_max + col_min - s)
        return df

    @staticmethod
    def subset_cols_num(cols, num, re_str):
        """
        subsets columns using column numbers
        :param cols: list of column names
        :param num: list of column numbers you want to subset
        :param re_str: regex to extract column numbers
        :return: subset columns
        """
        return cols[cols.str.extract(re_str)[0].astype(float).isin(num)]

    @staticmethod
    def is_valid_discrete(df, vals):
        """
        checks whether values in df are within range of discrete values or is missing
        :param df: pd.DataFrame or pd.Series
        :param vals: list of acceptable values
        :return: pd.DataFrame or pd.Series of dtype Bool
        """
        return ~df.isin(vals) & ~df.isna()

    @staticmethod
    def argwhere(df):
        """
        returns list of (index, colname) tuples where df is True
        :param df: pd.DataFrame or pd.Series of dtype Bool
        :return: pd.DataFrame where each row indicates indices where df is True
        """
        idx = np.argwhere(df)
        # dataframe
        if idx.shape[-1] == 2:
            idx = pd.DataFrame(idx, columns=['index', 'column'])
            idx['index'] = idx['index'].map(lambda x: df.index[x])
            idx['column'] = idx['column'].map(lambda x: df.columns[x])
        # series
        elif idx.shape[-1] == 1:
            idx = pd.DataFrame(idx, columns=['index'])
            idx['column'] = df.name
        else:
            raise ValueError('idx should be either pd.DataFrame of pd.Series')
        return idx

    @staticmethod
    def handle_duplicate(df, keep):
        """
        handles potential duplicate columns due to scoring
        :param df: pd.DataFrame of scored data
        :param keep: how to handle discrepancy
        :return: pd.DataFrame of scored data without duplicate columns
        """
        def drop_if_same(df, keep):
            assert len(df) <= 2
            # drop one if no discrepancy
            if df.eq(df.iloc[0, :], axis='columns').all().all():
                ser = df.iloc[0, :]
            # decide how to drop if discrepancy
            else:
                if keep:
                    ser = df.iloc[~df.index.duplicated(keep=keep), :].squeeze()
                else:
                    raise ExceptionWithData(
                        'Handling of duplicates undefined',
                        df.iloc[df.index.duplicated(keep=False), :]
                    )
            return ser
        df = df.T
        df = df.groupby(df.index).apply(lambda x: drop_if_same(x, keep))
        return df.T

    @classmethod
    def process(cls, df, output_path, mapping=None, to_na=True, rev_code=False, keep=None, **kwargs):
        """
        process the data
        :param df: DataFrame of data
        :param output_path: output path
        :param mapping: dict mapping df column names to expected measure column names
        :param to_na: bool whether to convert invalid values to np.nan
        :param rev_code: bool whether to reverse code
        :param keep: 'first' keeps the original score, 'last' keeps newly calculated score
        :return: DataFrame of relevant data + score
        """
        df = df.copy()
        # make sure index is in expected format
        if not (cls.get_index() == df.index.names).all():
            raise ValueError('Index names not expected')
        # rename columns to match expected
        if mapping:
            df = df.rename(columns=mapping)
        # subset to relevant columns depending on if score already included
        if len(cls.get_score_suffixes()) == 0:
            score_cols = []
        else:
            score_cols = [f"{cls.get_prefix()}_{x}" for x in cls.get_score_suffixes()]
        df = df.loc[:, [*cls.get_cols(), *df.columns[df.columns.isin(score_cols)]]]
        assert not df.columns.duplicated().any()
        # check if any outside of range
        idx = cls.check_range(df)
        # convert or raise
        if to_na:
            for row in idx.iterrows():
                df.loc[row['index'], row['column']] = np.nan
        elif len(idx) > 0:
            raise ExceptionWithData('Invalid range', idx)
        # reverse code
        if rev_code:
            df = cls.reverse_code(df, kwargs['col_num'], kwargs['re_str'], kwargs['col_min'], kwargs['col_max'])
        # score
        assert isinstance(df, pd.DataFrame)
        score = cls.score(df)
        df = pd.concat([df[cls.get_cols()], score], axis=1)
        assert isinstance(df, pd.DataFrame)
        # handle potential duplicate columns due to scoring
        df = cls.handle_duplicate(df, keep)
        # check
        assert df.columns.str.match(fr"^{cls.get_prefix()}_.+$").all()
        assert not df.columns.duplicated().any()
        # reorder
        df = df[[*cls.get_cols(), *df.columns[df.columns.isin(score_cols)]]]
        # save df
        df.to_csv(output_path)
        return df
