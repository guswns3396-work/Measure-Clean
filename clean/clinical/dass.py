from ..measure import Measure


import pandas as pd


# Depression Anxiety Stress Scale
# https://crossingborders.global/wp-content/uploads/2020/11/DASS-42-editable.pdf

class DASS42(Measure):
    @classmethod
    def get_prefix(cls):
        return 'dass42'

    @classmethod
    def get_score_suffixes(cls):
        return [f"score_{x}" for x in ['dep', 'anx', 'str']]

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(42)]

    @classmethod
    def check_range(cls, df):
        vals = [i for i in range(0, 3 + 1)]
        return cls.argwhere(cls.is_invalid_discrete(df[cls.get_cols()], vals))

    @classmethod
    def score(cls, df):
        dep = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
        anx = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
        str = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

        scored = pd.DataFrame([], columns=[f"{cls.get_prefix()}_{x}" for x in cls.get_score_suffixes()])
        for score, cols in zip(scored.columns, [dep, anx, str]):
            cols = cls.subset_cols_num(df.columns, cols, fr"{cls.get_prefix()}_(\d+)")
            scored[score] = df[cols].sum(axis=1, skipna=False)
        return scored
