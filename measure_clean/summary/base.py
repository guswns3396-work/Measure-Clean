from ..measure import Measure

import pandas as pd


class Base(Measure):
    @classmethod
    def get_prefix(cls):
        return 'base'

    @classmethod
    def get_score_suffixes(cls):
        raise NotImplementedError("Base class does not have scores")

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i}" for i in [
            'sex',
            'race',

        ]]

