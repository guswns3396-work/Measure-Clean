from .base import Base


class Medication(Measure):
    @classmethod
    def get_prefix(cls):
        return 'treatment'

    @classmethod
    def get_cols(cls):
        return [f"{cls.get_prefix()}_{i}" for i in [
            'current_treatment',
            'current_treatment_desc',
            ''
        ]]
