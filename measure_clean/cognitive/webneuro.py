from ..measure import Measure
from .parentneuro import ParentNeuro

import pandas as pd


# https://www.tandfonline.com/doi/full/10.1080/13803391003596496#appendixes

class WebNeuroCompatible(ParentNeuro):
    """
    webneuro measure that have variables corresponding to
    integneuro counterpart
    """

    @classmethod
    def get_prefix(cls):
        return 'webneuro'

    @classmethod
    def get_score_suffixes(cls):
        return []

    @classmethod
    def get_cols(cls):
        emotions = cls.get_emotions()
        suffixes = [
            # motor tapping
            'tdomnk',
            'tdomsdk',

            # choice reaction
            'chlrrtav',

            # verbal recall
            'ctmsco13',

            # explicit emotion
            *[f'getcp{i}' for i in emotions],
            *[f'getrt{i}' for i in emotions],

            # digt span
            'digitsp',
            'digitot',

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
            'esoadur1',
            'esoadur2',
            'esoaerr1',
            'esoaerr2',

            # go no go
            'g2avrtk',
            'g2fnk',
            'g2fpk',
            'g2sdrtk',
            'g2errk',

            # delayed verbal recall
            'ctmrec4',

            # implicit emotion
            *[f'dgtcrtn{i}' for i in emotions[:-1]],
            *[f'dgtrt{i}' for i in emotions],

            # working memory
            'wmfnk',
            'wmfpk',
            'wmrtk',
            'wmacck',

            # maze
            'emzcompk',
            'emzinitk',
            'emzoverk',
            'emzerrk',
            'emztrlsk'
        ]
        return [f"{cls.get_prefix()}_{i}" for i in suffixes] + \
               [f"{cls.get_prefix()}_{i}_norm" for i in suffixes]

    @classmethod
    def get_var_mapping(cls):
        mapping = [
            'tdomnk',
            'tdomsdk',
            'chlrrtav',
            'ctmsco13',
            'getcp',
            'getrt',
            'digitsp',
            'digitot',
            'vcrtne',
            'vi_sco',
            'vi_err',
            'vi_difrt',
            'scavr0t',
            'esoadur',
            'esoaerr',
            'g2avrtk',
            'g2fnk',
            'g2fpk',
            'g2errk',
            'g2sdrtk',
            'ctmrec4',
            'dgtrt',
            'dgtcn',
            'wmfnk',
            'wmfpk',
            'wmacck',
            'wmrtk',
            'emzcompk',
            'emzinitk',
            'emzoverk',
            'emzerrk',
            'emztrlsk'
        ]
        mapping = {k: k for k in mapping}
        return mapping
