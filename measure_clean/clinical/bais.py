from ..measure import Measure

import pandas as pd


# BAIS (Behavioral Activation and Behavioral Inhibition Scales (BAI))
# https://arc.psych.wisc.edu/self-report/behavioral-activation-and-behavioral-inhibition-scales-bai/

class BAIS(Measure):
    @classmethod
    def get_prefix(cls):
        return 'bais'

    @classmethod
    def get_score_suffixes(cls):
        return [f"score_{x}" for x in ['drive', 'fun', 'reward', 'bis']]

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(20)]

    @classmethod
    def get_restr(cls):
        return r"bais_(\d+)"

    @classmethod
    def get_min(cls):
        return 1

    @classmethod
    def get_max(cls):
        return 4

    @classmethod
    def check_range(cls, df):
        vals = [i for i in range(1, 4 + 1)]
        return cls.argwhere(cls.is_invalid_discrete(df, vals))

    @classmethod
    def score(cls, df):
        # reverse code for scoring
        rev_cols = [1, 18]
        df = cls.reverse_code(df, rev_cols, cls.get_restr(), 1, 4)

        # score
        drive = [2, 7, 9, 17]
        fun = [4, 8, 12, 16]
        reward = [3, 5, 11, 14, 19]
        bis = [1, 6, 10, 13, 15, 18, 20]

        scored = pd.DataFrame(
            [],
            columns=[f"{cls.get_prefix()}_{x}" for x in cls.get_score_suffixes()]
        )
        for score, cols in zip(scored.columns, [drive, fun, reward, bis]):
            cols = cls.subset_cols_num(df.columns, cols, cls.get_restr())
            scored[score] = df[cols].sum(axis=1, skipna=False)
        return scored
