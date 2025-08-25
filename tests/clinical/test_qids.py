import unittest
import numpy as np
import pandas as pd
from psypy.clean.clinical.qids import QIDS


class TestScore(unittest.TestCase):

    def setUp(self):
        self.TestMeasure = QIDS()

    def test__standard__score(self):
        np.random.seed(0)
        df = pd.DataFrame(
            np.random.randint(low=0, high=4, size=(5, 16)),
            columns=[f"{self.TestMeasure.get_prefix()}_{i + 1}" for i in range(16)]
        )
        target = np.array(
            [[3, 3, 3, 9, 2],
             [2, 2, 3, 4, 1],
             [3, 2, 3, 8, 3],
             [3, 1, 3, 10, 3],
             [3, 0, 2, 4, 2]]
        )
        target = np.sum(target, axis=1)
        source = self.TestMeasure.score(df)
        self.assertTrue((target == source).all().all())


if __name__ == "__main__":
    unittest.main()
