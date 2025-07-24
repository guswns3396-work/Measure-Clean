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
            *[f'getrt{i}' for i in emotions],

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
            'memrec7',

            # implicit emotion
            *[f'cdsgcn{i}' for i in emotions[:-1]],
            *[f'cdsgrt{i}' for i in emotions],

            # working memory
            'wmerr',
            'wmfn',
            'wmfp',
            'wmrt',

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
            'getrt': 'getrt',
            'digitsp': 'digitsp',
            'digitot': 'digitot',
            'vcrtne': 'vcrtne',
            'vi_sco': 'vi_sco',
            'vi_err': 'vi_err',
            'vi_difrt': 'vi_difrt',
            'scavr0t': 'scavr0t',
            'esoadur': 'swoadur',
            'esoaerr': 'swoaerr',
            'g2avrtk': 'gngavrt',
            'g2fnk': 'gngfn',
            'g2fpk': 'gngfp',
            'g2errk': 'gngerr',
            'g2sdrtk': 'gngsdrt',
            'ctmrec4': 'memrec7',
            'dgtrt': 'cdsgrt',
            'dgtcn': 'cdsgcn',
            'wmfnk': 'wmfn',
            'wmfpk': 'wmfp',
            'wmacck': 'wmerr',
            'wmrtk': 'wmrt',
            'emzcompk': 'emzcmpin',
            'emzinitk': 'emziniin',
            'emzoverk': 'emzovrin',
            'emzerrk': 'emzerrin',
            'emztrlsk': 'emztrlin'
        }

        return mapping
