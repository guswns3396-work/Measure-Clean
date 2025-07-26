from ..measure import Measure

import numpy as np
import pandas as pd


# Hamilton Rating Scale for Depression, also known as the Hamilton Depression Rating Scale, or HAM-D
# https://www.med.upenn.edu/cbti/assets/user-content/documents/Hamilton%20Rating%20Scale%20for%20Depression%20(HAM-D).pdf

class HDRS(Measure):
    @classmethod
    def get_score_suffixes(cls):
        return ['score']

    @classmethod
    def score(cls, df):
        df = df[[f"{cls.get_prefix()}_{i}" for i in range(1, 17 + 1)]]
        score = df.sum(axis=1, skipna=False)
        score.name = f"{cls.get_prefix()}_{cls.get_score_suffixes()[-1]}"
        return score

    @classmethod
    def check_range(cls, df):
        lik04 = [1, 2, 3, 7, 8, 9, 10, 11, 15, 19, 20]
        lik02 = [4, 5, 6, 12, 13, 14, 16, 17, 18, 21]
        idx = []
        for cols, vals in zip([lik04, lik02], [range(0, 4 + 1), range(0, 2 + 1)]):
            cols = cls.subset_cols_num(df.columns, cols, fr"{cls.get_prefix()}_(\d+)")
            idx.append(
                cls.argwhere(
                    cls.is_invalid_discrete(df[cols], [i for i in vals])
                )
            )
        idx = pd.concat(idx, axis=0)
        return idx


class HDRS21(HDRS):
    @classmethod
    def get_prefix(cls):
        return 'hdrs21'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(21)]


class HDRS17(HDRS):
    @classmethod
    def get_prefix(cls):
        return 'hdrs17'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(17)]
