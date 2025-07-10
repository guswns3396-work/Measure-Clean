from ..measure import Base

import pandas as pd


class Basic(Base):
    @classmethod
    def get_prefix(cls):
        return 'basic'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i}" for i in [
            'sex',
            'race',
            'phenotype',
            'treatment',
            'treatment_naive'
        ]]

    @classmethod
    def process(cls, df):
