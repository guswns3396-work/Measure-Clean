from ..measure import Measure


import pandas as pd


# Emotion Regulation Questionnaire
# https://www.carepatron.com/files/emotion-regulation-questionnaire.pdf

class ERQ(Measure):
    @classmethod
    def get_prefix(cls):
        return 'erq'

    @classmethod
    def get_score_suffixes(cls):
        return [f"score_{x}" for x in ['cog', 'sup']]

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(10)]

    @classmethod
    def check_range(cls, df):
        vals = [i for i in range(1, 7 + 1)]
        return cls.argwhere(cls.is_invalid_discrete(df[cls.get_cols()], vals))

    @classmethod
    def score(cls, df):
        cog = [1, 3, 5, 7, 8, 10]
        sup = [2, 6, 4, 9]

        scored = pd.DataFrame([], columns=[f"{cls.get_prefix()}_{x}" for x in cls.get_score_suffixes()])
        for score, cols in zip(scored.columns, [cog, sup]):
            cols = cls.subset_cols_num(df.columns, cols, fr'{cls.get_prefix()}_(\d+)')
            scored[score] = df[cols].sum(axis=1, skipna=False)
        return scored
