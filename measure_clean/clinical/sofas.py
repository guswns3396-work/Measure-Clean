from ..measure import Measure

import numpy as np
import pandas as pd


# Social and Occupational Functioning Assessment Scale

class SOFASRating(Measure):
    @classmethod
    def get_prefix(cls):
        return 'sofas'

    @classmethod
    def get_cols(cls):
        return [f"{self.prefix}_rating"]

    @classmethod
    def check_range(cls, df):
        vals = [i in range(0, 100 + 1)]
        return cls.argwhere(~df.isin(vals + [np.nan]))

    @classmethod
    def score(cls, df):
        return df


class SOFASCategory(Measure):
    @classmethod
    def get_prefix(cls):
        return 'sofas'

    @classmethod
    def get_cols(cls):
        return [f"{self.prefix}_category"]

    @classmethod
    def check_range(cls, df):
        vals = [i in range(0, 10 + 1)]
        return cls.argwhere(~df.isin(vals + [np.nan]))

    @classmethod
    def score(cls, df):
        return df
