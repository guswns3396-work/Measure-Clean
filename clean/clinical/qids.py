from ..measure import Measure

import numpy as np
import pandas as pd


# Quick Inventory of Depressive Symptomatology (QIDS)

class QIDS(Measure):
    """
    QIDS for when it's now known whether self-report or clinician
    """
    @classmethod
    def get_prefix(cls):
        return 'qids'

    @classmethod
    def get_score_suffixes(cls):
        return ['score']

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(16)]

    @classmethod
    def check_range(cls, df):
        vals = [i for i in range(0, 3 + 1)]
        return cls.argwhere(cls.is_invalid_discrete(df[cls.get_cols()], vals))

    @classmethod
    def score(cls, df):
        subscores = (
            # sleep subscore (items 1-4)
            np.nanmax(df[[fr"{cls.get_prefix()}_{i}" for i in range(1, 4 + 1)]], axis=1),
            # item 5
            df[fr"{cls.get_prefix()}_5"],
            # appetite / weight subscore (items 6-9)
            np.nanmax(df[[fr"{cls.get_prefix()}_{i}" for i in range(6, 9 + 1)]], axis=1),
            # items 10-14
            np.sum(df[[fr"{cls.get_prefix()}_{i}" for i in range(10, 14 + 1)]].values, axis=1),
            # psychomotor subscore
            np.nanmax(df[[fr"{cls.get_prefix()}_{i}" for i in range(15, 16 + 1)]], axis=1)
        )
        score = pd.Series(
            np.sum(np.vstack(subscores).T, axis=1),
            name=f"{cls.get_prefix()}_{cls.get_score_suffixes()[-1]}",
            index=df.index
        )
        return score
