from ..measure import Measure


import pandas as pd


# Patient Health Questionnaire (PHQ-9)
# https://med.stanford.edu/fastlab/research/imapp/msrs/_jcr_content/main/accordion/accordion_content3/download_256324296/file.res/PHQ9%20id%20date%2008.03.pdf

class PHQ7(Measure):
    @classmethod
    def get_prefix(cls):
        return 'phq9'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i + 1}" for i in range(9)]

    @classmethod
    def check_range(cls, df):
        vals = [i in range(0, 3 + 1)]
        return cls.argwhere(cls.is_valid_discrete(df, vals))

    @classmethod
    def score(cls, df):
        score = df.sum(axis=1, skipna=False)
        return score
