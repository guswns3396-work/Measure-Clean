from ..measure import Measure
from .parentneuro import ParentNeuro

import pandas as pd


# https://www.tandfonline.com/doi/full/10.1080/13803391003596496#appendixes

class IntegNeuroCompatible(ParentNeuro):
    """
    integneuro measure that have variables corresponding to
    webneuro counterpart
    """

    @classmethod
    def get_prefix(cls):
        return 'integneuro'

    @classmethod
    def get_score_suffixes(cls):
        return []

    @classmethod
    def get_cols(cls):
        emotions = cls.get_emotions()
        suffixes = [
            # motor tapping
            'tapdomn',
            'tapdomsd',

            # choice reaction
            'ch_avrt',

            # verbal recall
            'memtot14',

            # explicit emotion
            *[f'getcp{i}' for i in emotions],
            *[f'getcrt{i}' for i in emotions],

            # digt span
            'digitot',
            'digitsp',

            # verbal interference
            'vi_err1',
            'vi_err2',
            'vi_sco1',
            'vi_sco2',
            'vcrtne',
            'vcrtne2',
            'vi_difrt',

            # switching of attention
            'scavr0t1',
            'scavr0t2',
            'swoadur1',
            'swoadur2',
            'swoaerr1',
            'swoaerr2',

            # go no go
            'gngavrt',
            'gngerr',
            'gngfn',
            'gngfp',
            'gngsdrt',

            # delayed verbal recall
            'memtot7',

            # implicit emotion
            *[f'cdsgcn{i}' for i in emotions[:-1]],
            *[f'cdsgcrt{i}' for i in emotions],

            # working memory
            'wmerr',
            'wmfn',
            'wmfp',
            'wmrt',
            'wmsd'

            # maze
            'emzcmpin',
            'emzerrin',
            'emziniin',
            'emzovrin',
            'emztrlin'
        ]
        return [f"{cls.get_prefix()}_{i}" for i in suffixes] + \
               [f"{cls.get_prefix()}_{i}_norm" for i in suffixes]

    @classmethod
    def get_var_mapping(cls):
        mapping = {
            'tdomnk': 'tapdomn',
            'tdomsdk': 'tapdomsd',
            'chlrrtav': 'ch_avrt',
            'ctmsco13': 'memtot14',
            'getcp': 'getcp',
            'getcrt': 'getcrt',
            'digitsp': 'digitsp',
            'digitot': 'digitot',
            'vcrtne': 'vcrtne',
            'vi_sco': 'vi_sco',
            'vi_err': 'vi_err',
            'scavr0t': 'scavr0t',
            'esoadur': 'swoadur',
            'esoaerr': 'swoaerr',
            'g2avrtk': 'gngavrt',
            'g2fnk': 'gngfn',
            'g2fpk': 'gngfp',
            'g2sdrtk': 'gngsdrt',
            'ctmrec4': 'memtot7',
            'dgtcrt': 'cdsgcrt',
            'wmfnk': 'wmfn',
            'wmfpk': 'wmfp',
            'wmrtk': 'wmrt',
            'emzcompk': 'emzcmpin',
            'emzinitk': 'emziniin',
            'emzoverk': 'emzovrin',
            'emzerrk': 'emzovrin',
            'emztrlsk': 'emztrlin'
        }
        return mapping

    @classmethod
    def score(cls, df):
        """
        verifies summary measures (variable that reference other variables)
        :param df: pd.DataFrame of data
        :return: pd.DataFrame of scored summary variables
        """
        emotions = cls.get_emotions()
        scores = [
            # verbal interference
            (df[f"{cls.get_prefix()}_vcrtne2"] - df[f"{cls.get_prefix()}_vcrtne"]) \
            .rename(f"{cls.get_prefix()}_vi_difrt"),
            # go no go
            df[[f"{cls.get_prefix()}_gng{i}" for i in ['fn', 'fp']]].sum(axis=1) \
            .rename(f"{cls.get_prefix()}_gngerr"),
            # implicit emotion
            (df[[f"{cls.get_prefix()}_cdsgcrt{i}" for i in emotions[-1]]] - df[f"{cls.get_prefix()}_cdsgcrtN"]) \
            .rename(columns=[f"{cls.get_prefix()}_cdsgcn{i}" for i in emotions[-1]]),
            # working memory
            df[[f"{cls.get_prefix()}_wm{i}" for i in ['fn', 'fp']]].sum(axis=1) \
            .rename(f"{cls.get_prefix()}_wmerr"),
        ]
        scores = pd.concat(scores, axis=1)
        return scores
