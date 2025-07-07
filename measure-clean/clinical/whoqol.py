from ..measure import Measure

import numpy as np
import pandas as pd


# https://depts.washington.edu/seaqol/docs/WHOQOL-BREF%20and%20Scoring%20Instructions.pdf

class WHOQOL(Measure):
    @classmethod
    def get_prefix(cls):
        return 'whoqol'

    @classmethod
    def get_cols(cls):
        return [f"{self.prefix}_{i + 1}" for i in range(26)]

    @classmethod
    def check_range(cls, df):
        vals = [i in range(1, 5 + 1)]
        return np.argwhere(~df.isin(vals + [np.nan]))

    @classmethod
    def score(cls, df):
        # reverse code for scoring
        rev_cols = [3, 4, 26]
        df = cls.reverse_code(df, rev_cols, cls.get_restr(), 1, 5)

        # score
        phyhea = [3, 4, 10, 15, 16, 17, 18]
        psy = [5, 6, 7, 11, 19, 26]
        socrel = [20, 21, 22]
        env = [8, 9, 12, 13, 14, 23, 24, 25]

        scored = pd.DataFrame([], columns=['phyhea', 'psy', 'socrel', 'env'])
        for score, cols in zip(scored.columns, [phyhea, psy, socrel, env]):
            cols = cls.subset_cols_num(df.columns, cols, r"whoqol_(\d+)")
            scored[score] = df[cols].sum(axis=1, skipna=False)
        return scored
