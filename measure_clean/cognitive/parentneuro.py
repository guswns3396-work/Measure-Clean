from ..measure import Measure

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
            # recalled > 0
            # TODO

            # explicit emotion
            # check accuracy
            cls.argwhere(
                cls.is_valid_perc(df[[f"{cls.get_prefix()}_{var_mapping['getcp']}{i}" for i in emotions]])
            ),
            # check rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['getcrt']}{i}" for i in emotions]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['getcrt']}{i}" for i in emotions]].isna()))
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
            # check score > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_sco']}{i}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_sco']}{i}" for i in [1, 2]]].isna()))
            ),
            # check err > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_err']}{i}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_err']}{i}" for i in [1, 2]]].isna()))
            ),

            # TODO

            # switching of attention
            # connection time > 0
            # duration > 0
            # 0 <= errors <= 25

            # go no go
            # rt > 0
            # fp, fn > 0
            # sd > 0

            # delayed recall
            # recalled > 0

            # implicit emotion
            # rt > 0

            # working memory
            # fn, fp > 0
            # rt > 0

            # maze
            # comp time > 0
            # init time > 0
            # over > 0
            # err > 0
            # trials > 2

        ]
        return pd.concat(idx, axis=0)
