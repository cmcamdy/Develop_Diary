import unittest
import numpy as np
import parameterize
import scipy.stats
# from distribution import config
from paddle.distribution import kl

import config
import lkj_cholesky
import paddle
import numbers

# paddle.enable_static()

np.random.seed(2024)
paddle.seed(2024)


def matrix_to_tril_vec(x, diagonal=-1):
    pass


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'dim', 'concentration', 'sample_method'),
    [
        ('2-onion', 2, 1.0, "onion"),
        ('2-cvine', 2, 1.0, "cvine"),
        ('3-onion', 3, 1.0, "onion"),
        ('3-cvine', 3, 1.0, "cvine")
    ]
)
class TestLKJCholesky(unittest.TestCase):
    def setUp(self):
        dim = self.dim
        concentration = self.concentration
        sample_method = self.sample_method
            
        self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(dim, concentration, sample_method)
                
    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': () + (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim),
            },
            {
                'input': (4, 2),
                'expect': (4, 2) + (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim),
            },
        ]
        for case in cases:
            self.assertTrue(tuple(self._paddle_lkj_cholesky.sample(case.get('input')).shape) == case.get('expect'))

           
    def test_log_prob(self):
        value = np.random.rand(self._paddle_lkj_cholesky.dim)
        print(value)
        sample = self._paddle_lkj_cholesky.sample()
        log_prob = self._paddle_lkj_cholesky.log_prob(sample)
        sample_tril = matrix_to_tril_vec(sample, diagonal = -1)


if __name__ == '__main__':
    unittest.main()
