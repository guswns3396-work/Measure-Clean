from ..measure import Measure


import pandas as pd


# GAD-7 (General Anxiety Disorder-7)

class GAD7(Measure):
    @classmethod
    def get_prefix(cls):
        return 'gad7'

    @classmethod
    def get_score_suffixes(cls):
        return ['score']

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(7)]

    @classmethod
    def check_range(cls, df):
        vals = [i for i in range(0, 3 + 1)]
        return cls.argwhere(cls.is_valid_discrete(df, vals))

    @classmethod
    def score(cls, df):
        score = df.sum(axis=1, skipna=False)
        score.name = f"{cls.get_prefix()}_{cls.get_score_suffixes()[-1]}"
        return score
