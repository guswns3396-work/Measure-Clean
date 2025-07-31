from ..measure import Base
import pandas as pd


class Summary(Base):
    @classmethod
    def get_prefix(cls):
        return 'summary'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i}" for i in [
            'sex',
            'race',
            'phenotype',
            'treatment',
            'treatment_naive',
            'dataset',

            'current_medication',
            'current_medication_desc',
            'past_medication',
            'past_medication_desc',

            'mdd_current',
            'bipolar_current',
            'gad_current',
            'panic_current',
            'social_phobia_current',
            'ptsd_current',
            'ocd_current',
            'anorexia_current',
            'other_eatdisorder_current'
        ]]

    @classmethod
    def check_range(cls, df):
        idx = [
            cls.argwhere(cls.is_invalid_discrete(df[f"{cls.get_prefix()}_sex"], ['M', 'F', 'O'])),
            cls.argwhere(cls.is_invalid_discrete(
                df[[f"{cls.get_prefix()}_{x}" for x in [
                    'treatment_naive',
                    'current_medication',
                    'past_medication',
                    'mdd_current',
                    'bipolar_current',
                    'gad_current',
                    'panic_current',
                    'social_phobia_current',
                    'ptsd_current',
                    'ocd_current',
                    'anorexia_current',
                    'other_eatdisorder_current'
                ]]],
                [1, 0]
            )),
            cls.argwhere(~df[f"{cls.get_prefix()}_race"].isin([
                'American Indian/Alaska Native',
                'Asian',
                'Native Hawaiian or Other Pacific Islander',
                'Black or African American',
                'White',
                'More Than One Race',
                'Other',
                'Unknown / Not Reported'
            ]))
        ]
        return pd.concat(idx, axis=0)
