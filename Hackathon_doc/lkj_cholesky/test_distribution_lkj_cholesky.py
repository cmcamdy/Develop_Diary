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


def lkj_cholesky_onion():
    pass


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'dim', 'concentration', 'sample_method'),
    [
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
                
    def test_shape(self):
        # print(self._paddle_lkj_cholesky.dim)
        # print((self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim))
        cases = [
            # {
            #     'input': (),
            #     'expect': () + (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim),
            # },
            {
                'input': (4, 2),
                'expect': (4, 2) + (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim),
            },
        ]
        for case in cases:
            self.assertTrue(tuple(self._paddle_lkj_cholesky.sample(case.get('input')).shape) == case.get('expect'))
            
           
    # def test_variance(self):
    #     with paddle.base.dygraph.guard(self.place):
    #         np.testing.assert_allclose(
    #             self._paddle_chi2.variance,
    #             scipy.stats.chi2.var(self.df),
    #             rtol=config.RTOL.get(
    #                 str(self._paddle_chi2.df.numpy().dtype)
    #             ),
    #             atol=config.ATOL.get(
    #                 str(self._paddle_chi2.df.numpy().dtype)
    #             ),
    #         )
           
    # def test_entropy(self):
    #     with paddle.base.dygraph.guard(self.place):
    #         np.testing.assert_allclose(
    #             self._paddle_chi2.entropy(),
    #             scipy.stats.chi2.entropy(self.df),
    #             rtol=config.RTOL.get(str(self.df.dtype)),
    #             atol=config.ATOL.get(str(self.df.dtype)),
    #         )

    # def test_prob(self):
    #     value = np.random.rand(*self._paddle_chi2.df.shape)
    #     with paddle.base.dygraph.guard(self.place):
    #         np.testing.assert_allclose(
    #             self._paddle_chi2.prob(paddle.to_tensor(value)),
    #             scipy.stats.chi2.pdf(value, self.df),
    #             rtol=config.RTOL.get(str(self.df.dtype)),
    #             atol=config.ATOL.get(str(self.df.dtype)),
    #         )

    # def test_log_prob(self):
    #     value = np.random.rand(*self._paddle_chi2.df.shape)
    #     with paddle.base.dygraph.guard(self.place):
    #         np.testing.assert_allclose(
    #             self._paddle_chi2.log_prob(paddle.to_tensor(value)),
    #             scipy.stats.chi2.logpdf(value, self.df),
    #             rtol=config.RTOL.get(str(self.df.dtype)),
    #             atol=config.ATOL.get(str(self.df.dtype)),
    #         )


if __name__ == '__main__':
    unittest.main()
