import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ExceptionWithData(Exception):
    def __init__(self, message, data):
        super().__init__(message)
        self.data = data


class Base(ABC):
    @classmethod
    @abstractmethod
    def get_prefix(cls):
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
    def process(cls, df, output_path, calc_age=None, mapping=None, to_na=True, rev_code=False, **kwargs):
        """
        :param df: DataFrame of data
        :param output_path: output path
        :param calc_age: whether to calculate age
        :param mapping: dict mapping df column names to expected measure column names
        :param to_na: bool whether to convert invalid values to np.nan
        :param rev_code: bool whether to reverse code
       """
        df = df.copy()
        # make sure index is in expected format
        if not (cls.get_index() == df.index.names).all():
            raise ValueError('Index names not expected')
        # rename columns to match expected
        if mapping:
            df = df.rename(columns=mapping)

        if calc_age is not None:
            if 'format' in kwargs:
                df = cls.calculate_age(df, how=calc_age, dob_col=kwargs['dob_col'],
                                       date_col=kwargs['date_col'], format=kwargs['format'])
            else:
                df = cls.calculate_age(df, how=calc_age, dob_col=kwargs['dob_col'], date_col=kwargs['date_col'])

        # subset to relevant columns
        df = cls.subset_relevant_cols(df)

        # drop all NaNs
        df = df[~df.isna().all(axis=1)]

        assert not df.columns.duplicated().any()
        # check if any outside of range
        idx = cls.check_range(df)
        idx = idx.drop_duplicates()
        # convert or raise
        if to_na == 'ignore':
            pass
        elif to_na:
            for _, row in idx.iterrows():
                df.loc[row['index'], row['column']] = np.nan
        elif len(idx) > 0:
            raise ExceptionWithData('Invalid range', idx)

        # reverse code
        if rev_code:
            df = cls.reverse_code(df, kwargs['col_num'], cls.get_restr(), cls.get_min(), cls.get_max())

        # score
        if 'keep' in kwargs:
            df = cls.score_if_needed(df, keep=kwargs['keep'])
        else:
            df = cls.score_if_needed(df, keep=None)

        # check
        assert df.columns.str.match(fr"^{cls.get_prefix()}_.+$").all()
        assert not df.columns.duplicated().any()

        # reorder
        df = cls.reorder(df)

        # save df
        if output_path is not None:
            df.to_csv(os.path.join(output_path, f"{cls.get_prefix()}.csv"))
        return df

    @classmethod
    def subset_relevant_cols(cls, df):
        df = df.loc[:, cls.get_cols()]
        return df

    @classmethod
    def score_if_needed(cls, df, keep):
        return df

    @classmethod
    def reorder(cls, df):
        df = df.loc[:, cls.get_cols()]
        return df

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
        df = df.reset_index()
        df.loc[:, cols] = df.loc[:, cols].apply(lambda s: col_max + col_min - s)
        df = df.set_index(['ID', 'SES', 'AGE'])
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
    def is_invalid_discrete(df, vals):
        """
        checks whether values in df are within range of discrete values or is missing
        returns true if invalid
        :param df: pd.DataFrame or pd.Series
        :param vals: list of acceptable values
        :return: pd.DataFrame or pd.Series of dtype Bool
        """
        return ~df.isin(vals) & ~df.isna()

    @staticmethod
    def argwhere(df):
        """
        returns list of (index, colname) tuples where df is True
        :param df: pd.DataFrame dtype Bool
        :return: pd.DataFrame where each row indicates indices where df is True
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        idx = np.argwhere(df)
        idx = pd.DataFrame(idx, columns=['index', 'column'])
        idx['index'] = idx['index'].map(lambda x: df.index[x])
        idx['column'] = idx['column'].map(lambda x: df.columns[x])
        return idx

    @staticmethod
    def calculate_age(df, how, dob_col, date_col, **kwargs):
        """
        calculates age based on dob and date of assessment
        :param df: dataframe
        :param how: whether to fill age or replace age
        :param dob_col: column for dob
        :param date_col: string for date column name or dict mapping session to date column
        :return: df with age
        """
        # unique ID <-> DOB
        dob = df.reset_index().set_index('ID')[dob_col].dropna().rename('_dob')
        assert not dob.index.duplicated().any()

        # date on different row as data => get to same row
        if isinstance(date_col, dict):
            dates = []
            for k in date_col:
                # unique ID <-> date
                date = df.reset_index().set_index('ID')[date_col[k]].rename('_date').dropna().to_frame()
                assert not date.index.duplicated().any()
                date['SES'] = k
                dates.append(date)
            # unique ID,SES <-> date
            dates = pd.concat(dates, axis=0).reset_index().set_index(['ID', 'SES'])['_date']
            assert not dates.index.duplicated().any()

            df = df.reset_index().set_index(['ID', 'SES'])
            assert '_date' not in df.columns
            df = df.merge(dates, how='left', left_index=True, right_index=True)
            df = df.reset_index().set_index('ID')

        # same row as data
        elif isinstance(date_col, str):
            assert '_date' not in df.columns
            df['_date'] = df[date_col]
            df = df.reset_index().set_index('ID')

        else:
            raise ValueError("Invalid 'date_col' argument")

        # get dob for each date
        assert '_dob' not in df.columns
        assert '_age' not in df.columns
        df = df.merge(dob, how='left', left_index=True, right_index=True)

        # calculate age
        if 'format' in kwargs:
            df['_age'] = np.floor((pd.to_datetime(df['_date'], format=kwargs['format']) - pd.to_datetime(df['_dob'], format=kwargs[
                'format'])) / pd.Timedelta(days=365.25))
        else:
            df['_age'] = np.floor(
                (pd.to_datetime(df['_date']) - pd.to_datetime(df['_dob'])) / pd.Timedelta(
                    days=365.25))

        if how == 'replace':
            df['AGE'] = df['_age']
        elif how == 'fill':
            df['AGE'] = df['AGE'].fillna(df['_age'])
        else:
            raise ValueError("Invalid 'how' argument")

        df = df.drop(columns=['_date', '_dob', '_age'])

        df = df.reset_index().set_index(['ID', 'SES', 'AGE'])

        return df


class Measure(Base):
    @classmethod
    @abstractmethod
    def get_score_suffixes(cls):
        """
        :return: list of suffixes for score variable name
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

    @classmethod
    def get_score_cols(cls):
        if len(cls.get_score_suffixes()) == 0:
            return []
        else:
            return [f"{cls.get_prefix()}_{x}" for x in cls.get_score_suffixes()]

    @classmethod
    def subset_relevant_cols(cls, df):
        # subset to relevant columns depending on if score already included
        df = df.loc[:, [*cls.get_cols(), *df.columns[df.columns.isin(cls.get_score_cols())]]]
        return df

    @classmethod
    def score_if_needed(cls, df, keep):
        # score
        assert isinstance(df, pd.DataFrame)
        score = cls.score(df)
        df = pd.concat([df[[*cls.get_cols(), *df.columns[df.columns.isin(cls.get_score_cols())]]], score], axis=1)
        assert isinstance(df, pd.DataFrame)
        # handle potential duplicate columns due to scoring
        df = cls.handle_duplicate(df, keep)
        return df

    @classmethod
    def reorder(cls, df):
        df = df[[*cls.get_cols(), *df.columns[df.columns.isin(cls.get_score_cols())]]]
        return df

    @staticmethod
    def handle_duplicate(df, keep):
        """
        handles potential duplicate columns due to scoring
        :param df: pd.DataFrame of scored data
        :param keep: how to handle discrepancy
        'first' keeps the original score, 'last' keeps newly calculated score
        :return: pd.DataFrame of scored data without duplicate columns
        """

        def drop_if_same(df, keep):
            assert len(df) <= 2
            # drop one if no discrepancy
            if (df.eq(df.iloc[0, :], axis='columns') | df.isna()).all().all():
                ser = df.iloc[0, :]
            # decide how to drop if discrepancy
            else:
                if keep:
                    ser = df.iloc[~df.index.duplicated(keep=keep), :].squeeze()
                else:
                    raise ExceptionWithData(
                        'Handling of duplicates undefined',
                        df
                    )
            return ser

        df = df.T
        df = df.groupby(df.index).apply(lambda x: drop_if_same(x, keep))
        return df.T
