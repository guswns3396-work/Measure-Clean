from ..measure import Measure

import numpy as np
import pandas as pd

# Brief Risk-Resilience Index for Screening
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3489810/

class BRISC15(Measure):
    @classmethod
    def get_prefix(cls):
        return 'brisc'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(15)]

    @classmethod
    def check_range(cls, df):
        vals = [i in range(1, 5 + 1)]
        return np.argwhere(~df.isin(vals + [np.nan]))

    @classmethod
    def score(cls, df):
        mapping = {
            'neg': [1, 2, 3, 4, 5],
            'emo': [6, 7, 8, 9, 10],
            'soc': [11, 12, 13, 14, 15]
        }

        scores = []
        for score in mapping:
            cols = mapping[score]
            scores.append(
                df[cls.subset_cols_num(df.columns, cols, r"brisc_(\d+)")].sum(axis=1, skipna=False)
            )
        return pd.concat(scores, axis=1)
