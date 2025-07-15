from ..measure import Measure
from abc import abstractmethod
import pandas as pd


class ParentNeuro(Measure):
    @classmethod
    @abstractmethod
    def get_var_mapping(cls):
        """
        maps variable name for check_range()
        :return: dict of variable mapping
        """
        pass

    @classmethod
    def get_emotions(cls):
        return "ADFHSN"

    @classmethod
    def is_valid_digitsp(cls, ser):
        """
        :param ser: pd.Series for digitsp
        :return: pd.Series of dtype Bool for valid values
        """
        return cls.is_valid_discrete(ser, [i for i in range(3, 9 + 1)])

    @classmethod
    def is_valid_digitot(cls, ser):
        """
        :param ser: pd.Series for digitot
        :return: pd.Series of dtype Bool for valid values
        """
        return cls.is_valid_discrete(ser, [i for i in range(0, 14 + 1)])

    @classmethod
    def is_valid_perc(cls, df):
        """
        :param df: pd.Series or pd.DataFrame of percent variables
        :return: pd.Series or pd.DataFrame of dtype Bool for valid values
        """
        return ((df <= 100) & (df >= 0)) | df.isna()

    @classmethod
    def check_range(cls, df):
        emotions = cls.get_emotions()
        var_mapping = cls.get_var_mapping()
        idx = [
            # motor tapping
            # check number of taps > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['tdomnk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['tdomnk']}"].isna()))
            ),
            # check sd > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['tdomsdk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['tdomsdk']}"].isna()))
            ),

            # choice reaction
            # check rt > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['chlrrtav']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['chlrrtav']}"].isna()))
            ),

            # verbal recall
            # recalled >= 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['ctmsco13']}"] >= 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['ctmsco13']}"].isna()))
            ),

            # explicit emotion
            # check accuracy
            cls.argwhere(
                cls.is_valid_perc(df[[f"{cls.get_prefix()}_{var_mapping['getcp']}{i}" for i in emotions]])
            ),
            # check rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['getrt']}{i}" for i in emotions]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['getrt']}{i}" for i in emotions]].isna()))
            ),

            # digit span
            # check span range
            cls.argwhere(cls.is_valid_digitsp(df[f"{cls.get_prefix()}_{var_mapping['digitsp']}"])),
            # check total range
            cls.argwhere(cls.is_valid_digitot(df[f"{cls.get_prefix()}_{var_mapping['digitot']}"])),

            # verbal interference
            # check rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vcrtne']}{i}" for i in ['', '2']]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vcrtne']}{i}" for i in ['', '2']]].isna()))
            ),
            # check score >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_sco']}{i}" for i in [1, 2]]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_sco']}{i}" for i in [1, 2]]].isna()))
            ),
            # check err >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_err']}{i}" for i in [1, 2]]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_err']}{i}" for i in [1, 2]]].isna()))
            ),

            # switching of attention
            # connection time > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['scavr0t']}{i}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['scavr0t']}{i}" for i in [1, 2]]].isna()))
            ),
            # duration > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['esoadur']}{i}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['esoadur']}{i}" for i in [1, 2]]].isna()))
            ),
            # 0 <= errors <= 25
            cls.argwhere(
                ~(((df[[f"{cls.get_prefix()}_{var_mapping['esoaerr']}{i}" for i in [1, 2]]] >= 0)
                  & (df[[f"{cls.get_prefix()}_{var_mapping['esoaerr']}{i}" for i in [1, 2]]] <= 25))
                  | (df[[f"{cls.get_prefix()}_{var_mapping['esoaerr']}{i}" for i in [1, 2]]].isna()))
            ),

            # go no go
            # rt > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['g2avrtk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['g2avrtk']}"].isna()))
            ),
            # fp, fn >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping[i]}" for i in ['g2fnk', 'g2fpk']]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping[i]}" for i in ['g2fnk', 'g2fpk']]].isna()))
            ),
            # sd > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['g2sdrtk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['g2sdrtk']}"].isna()))
            ),

            # delayed recall
            # recalled >= 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['ctmrec4']}"] >= 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['ctmrec4']}"].isna()))
            ),

            # implicit emotion
            # rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['dgtcrt']}{i}" for i in emotions]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['dgtcrt']}{i}" for i in emotions]].isna()))
            ),

            # working memory
            # fn, fp >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping[i]}" for i in ['wmfnk', 'wmfpk']]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping[i]}" for i in ['wmfnk', 'wmfpk']]].isna()))
            ),
            # rt > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['wmrtk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['wmrtk']}"].isna()))
            ),

            # maze
            # comp time > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['emzcompk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['emzcompk']}"].isna()))
            ),
            # init time > 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['emzinitk']}"] > 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['emzinitk']}"].isna()))
            ),
            # over >= 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['emzoverk']}"] >= 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['emzoverk']}"].isna()))
            ),
            # err >= 0
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['emzerrk']}"] >= 0)
                  | (df[f"{cls.get_prefix()}_{var_mapping['emzerrk']}"].isna()))
            ),
            # trials >= 2
            cls.argwhere(
                ~((df[f"{cls.get_prefix()}_{var_mapping['emztrlsk']}"] >= 2)
                  | (df[f"{cls.get_prefix()}_{var_mapping['emztrlsk']}"].isna()))
            ),
            # maze comp time > maze init time
            cls.argwhere((df[f"{cls.get_prefix()}_{var_mapping['emzcompk']}"]
                          > df[f"{cls.get_prefix()}_{var_mapping['emzinitk']}"]).rename('emzcompk')),
            # emzerr > emzover
            cls.argwhere((df[f"{cls.get_prefix()}_{var_mapping['emzerrk']}"]
                          > df[f"{cls.get_prefix()}_{var_mapping['emzoverk']}"]).rename('emzerrk'))
        ]
        return pd.concat(idx, axis=0)
