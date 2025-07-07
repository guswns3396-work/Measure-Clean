from abc import ABC, abstractmethod
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
    def get_cols(cls):
        pass

    @classmethod
    @abstractmethod
    def check_range(cls, df):
        """
        checks range and returns indices of invalid values
        :param df: DataFrame of data
        :return: indices of invalid values
        """
        pass

    @classmethod
    @abstractmethod
    def score(cls, df):
        """
        scores the measure
        :param df: DataFrame of data (should match the expected coding of scoring guide)
        :return: pd.Series of scores
        """
        pass

    @staticmethod
    def get_index():
        return pd.Index(['ID', 'SES'])

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

    @classmethod
    def process(cls, df, output_path, to_na=True, mapping=None, rev_code=False, **kwargs):
        """
        process the data
        :param df: DataFrame of data
        :param output_path: output path
        :param to_na: bool whether to convert invalid values to np.nan
        :param mapping: dict mapping df column names to expected measure column names
        :param rev_code: bool whether to reverse code
        :return: DataFrame of relevant data + score
        """
        df = df.copy()
        # make sure index is in expected format
        if not (cls.get_index() == df.index.names).all():
            raise ValueError('Index names not expected')
        # rename columns to match expected
        if mapping:
            df = df.rename(columns=mapping)
        # make sure columns for measure are in df
        if not cls.get_cols().isin(df.columns).all():
            raise ExceptionWithData(
                'Expected columns not found',
                cls.get_cols()[~cls.get_cols().isin(df.columns)]
            )
        # subset to relevant columns
        df = df.loc[:, cls.get_cols()]
        # check if any outside of range
        idx = cls.check_range(df)
        # convert or raise
        if to_na:
            for (i, j) in idx:
                df.iloc[i, j] = np.nan
        else:
            if idx:
                raise ExceptionWithData('Invalid range', idx)
        # reverse code
        if rev_code:
            df = cls.reverse_code(df, kwargs['col_num'], kwargs['re_str'], kwargs['col_min'], kwargs['col_max'])
        # score
        score = cls.score(df)
        # save df
        df = pd.concat([df[cls.get_cols()], score], axis=1)
        df.to_csv(output_path)
        return df
