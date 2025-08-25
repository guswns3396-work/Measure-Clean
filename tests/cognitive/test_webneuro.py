import unittest
import numpy as np
import pandas as pd
from clean.cognitive.webneuro import WebNeuroCompatible


class TestProcess(unittest.TestCase):

    def setUp(self):
        self.TestMeasure = WebNeuroCompatible()

    def test__standard__process(self):
        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
