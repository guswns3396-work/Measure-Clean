import unittest
import numpy as np
import pandas as pd
from measure_clean.measure import Measure
from measure_clean.clinical.hdrs import HDRS21, HDRS17


# TODO: test other versions of HDRS than HDRS21
class TestCheckRange(unittest.TestCase):

    def setUp(self):
        self.TestMeasure = HDRS21()

    def test__standard__check_range(self):
        np.random.seed(0)
        lik04 = [1, 2, 3, 7, 8, 9, 10, 11, 15, 19, 20]
        lik02 = [4, 5, 6, 12, 13, 14, 16, 17, 18, 21]
        df1 = pd.DataFrame(
            np.random.randint(low=0, high=6, size=(5, len(lik04))),
            columns=[f"{self.TestMeasure.get_prefix()}_{i}" for i in lik04]
        )
        df2 = pd.DataFrame(
            np.random.randint(low=0, high=4, size=(5, len(lik02))),
            columns=[f"{self.TestMeasure.get_prefix()}_{i}" for i in lik02]
        )
        df = pd.concat([df1, df2], axis=1)
        tmp = df[[f"{self.TestMeasure.get_prefix()}_{i}" for i in range(1, 21 + 1)]]
        target = [[0, 1], [3, 1], [2, 3], [1, 4], [0, 5], [4, 5], [2, 8], [1, 10],
                  [2, 11], [3, 11], [1, 12], [3, 12], [4, 12], [0, 14], [4, 14], [1, 15], [3, 15], [4, 15],
                  [0, 16], [1, 16], [4, 16], [1, 18], [4, 18], [2, 20]]
        source = self.TestMeasure.check_range(df)
        target = set([
            tuple([tmp.index[x[0]], tmp.columns[x[1]]])
            for x in target
        ])
        source = set([tuple([x['index'], x['column']]) for _, x in source.iterrows()])
        self.assertEqual(target, source)


class TestScore(unittest.TestCase):

    def setUp(self):
        self.TestMeasure = HDRS21()

    def test__standard__score(self):
        np.random.seed(0)
        lik04 = [1, 2, 3, 7, 8, 9, 10, 11, 15, 19, 20]
        lik02 = [4, 5, 6, 12, 13, 14, 16, 17, 18, 21]
        df1 = pd.DataFrame(
            np.random.randint(low=0, high=5, size=(5, len(lik04))),
            columns=[f"{self.TestMeasure.get_prefix()}_{i}" for i in lik04]
        )
        df2 = pd.DataFrame(
            np.random.randint(low=0, high=3, size=(5, len(lik02))),
            columns=[f"{self.TestMeasure.get_prefix()}_{i}" for i in lik02]
        )
        df = pd.concat([df1, df2], axis=1)
        target = [34, 19, 25, 27, 26]
        source = self.TestMeasure.score(df)

        self.assertTrue((target == source).all())


if __name__ == "__main__":
    unittest.main()
