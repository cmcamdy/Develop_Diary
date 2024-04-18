import unittest
import numpy as np
import parameterize
import scipy.stats
from distribution import config


from chi2 import Chi2

paddle.enable_static()

np.random.seed(2023)
paddle.seed(2023)
@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df'),
    [
        ()
    ]
)
class TestGamma(unittest.TestCase):