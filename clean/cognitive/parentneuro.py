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
    def is_invalid_digitsp(cls, ser):
        """
        :param ser: pd.Series for digitsp
        :return: pd.Series of dtype Bool for valid values
        """
        return cls.is_invalid_discrete(ser, [i for i in range(3, 9 + 1)])

    @classmethod
    def is_invalid_digitot(cls, ser):
        """
        :param ser: pd.Series for digitot
        :return: pd.Series of dtype Bool for valid values
        """
        return cls.is_invalid_discrete(ser, [i for i in range(0, 14 + 1)])

    @classmethod
    def is_invalid_perc(cls, df):
        """
        :param df: pd.Series or pd.DataFrame of percent variables
        :return: pd.Series or pd.DataFrame of dtype Bool for valid values
        """
        return ~(((df <= 100) & (df >= 0)) | df.isna())

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
                cls.is_invalid_perc(df[[f"{cls.get_prefix()}_{var_mapping['getcp' + i]}" for i in emotions]])
            ),
            # check rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['getrt' + i]}" for i in emotions]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['getrt' + i]}" for i in emotions]].isna()))
            ),

            # digit span
            # check span range
            cls.argwhere(cls.is_invalid_digitsp(df[f"{cls.get_prefix()}_{var_mapping['digitsp']}"])),
            # check total range
            cls.argwhere(cls.is_invalid_digitot(df[f"{cls.get_prefix()}_{var_mapping['digitot']}"])),

            # verbal interference
            # check rt > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vcrtne' + i]}" for i in ['', '2']]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vcrtne' + i]}" for i in ['', '2']]].isna()))
            ),
            # check score >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_sco' + str(i)]}" for i in [1, 2]]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_sco' + str(i)]}" for i in [1, 2]]].isna()))
            ),
            # check err >= 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['vi_err' + str(i)]}" for i in [1, 2]]] >= 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['vi_err' + str(i)]}" for i in [1, 2]]].isna()))
            ),

            # switching of attention
            # connection time > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['scavr0t' + str(i)]}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['scavr0t' + str(i)]}" for i in [1, 2]]].isna()))
            ),
            # duration > 0
            cls.argwhere(
                ~((df[[f"{cls.get_prefix()}_{var_mapping['esoadur' + str(i)]}" for i in [1, 2]]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['esoadur' + str(i)]}" for i in [1, 2]]].isna()))
            ),
            # 0 <= errors <= 25
            cls.argwhere(
                ~(((df[[f"{cls.get_prefix()}_{var_mapping['esoaerr' + str(i)]}" for i in [1, 2]]] >= 0)
                   & (df[[f"{cls.get_prefix()}_{var_mapping['esoaerr' + str(i)]}" for i in [1, 2]]] <= 25))
                  | (df[[f"{cls.get_prefix()}_{var_mapping['esoaerr' + str(i)]}" for i in [1, 2]]].isna()))
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
                ~((df[[f"{cls.get_prefix()}_{var_mapping['dgtrt' + i]}" for i in emotions]] > 0)
                  | (df[[f"{cls.get_prefix()}_{var_mapping['dgtrt' + i]}" for i in emotions]].isna()))
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
            cls.argwhere(~((df[f"{cls.get_prefix()}_{var_mapping['emzcompk']}"]
                            >= df[f"{cls.get_prefix()}_{var_mapping['emzinitk']}"]) |
                           df[[f"{cls.get_prefix()}_{var_mapping['emzcompk']}",
                               f"{cls.get_prefix()}_{var_mapping['emzinitk']}"]].isna().any(axis=1))
                         .rename(f"{cls.get_prefix()}_{var_mapping['emzcompk']}")),
            # emzerr > emzover
            cls.argwhere(~((df[f"{cls.get_prefix()}_{var_mapping['emzerrk']}"]
                           >= df[f"{cls.get_prefix()}_{var_mapping['emzoverk']}"]) |
                           df[[f"{cls.get_prefix()}_{var_mapping['emzerrk']}",
                               f"{cls.get_prefix()}_{var_mapping['emzoverk']}"]].isna().any(axis=1))
                         .rename(f"{cls.get_prefix()}_{var_mapping['emzerrk']}")),
        ]
        return pd.concat(idx, axis=0)

    @classmethod
    def score(cls, df):
        """
        verifies summary measures (variable that reference other variables)
        :param df: pd.DataFrame of data
        :return: pd.DataFrame of scored summary variables
        """
        emotions = cls.get_emotions()
        var_mapping = cls.get_var_mapping()
        scores = [
            # verbal interference
            (df[f"{cls.get_prefix()}_{var_mapping['vcrtne2']}"] - df[f"{cls.get_prefix()}_{var_mapping['vcrtne']}"]) \
                .rename(f"{cls.get_prefix()}_{var_mapping['vi_difrt']}"),
            # go no go
            df[[f"{cls.get_prefix()}_{var_mapping['g2' + i + 'k']}" for i in ['fn', 'fp']]].sum(axis=1, skipna=False) \
                .rename(f"{cls.get_prefix()}_{var_mapping['g2errk']}"),
            # implicit emotion
            (df[[f"{cls.get_prefix()}_{var_mapping['dgtrt' + i]}" for i in emotions[:-1]]].apply(
                lambda s: s - df[f"{cls.get_prefix()}_{var_mapping['dgtrtN']}"]
            )).rename(columns={
                f"{cls.get_prefix()}_{var_mapping['dgtrt' + i]}": f"{cls.get_prefix()}_{var_mapping['dgtcn' + i]}"
                for i in emotions[:-1]
            }),
            # working memory
            df[[f"{cls.get_prefix()}_{var_mapping['wm' + i + 'k']}" for i in ['fn', 'fp']]].sum(axis=1, skipna=False) \
                .rename(f"{cls.get_prefix()}_{var_mapping['wmacck']}"),
        ]
        scores = pd.concat(scores, axis=1)
        return scores
