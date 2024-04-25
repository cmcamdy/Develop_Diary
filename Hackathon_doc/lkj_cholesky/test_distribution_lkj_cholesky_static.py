import unittest
import numpy as np
import parameterize
import scipy.stats
# from distribution import config
from paddle.distribution import kl

import config
import lkj_cholesky
from lkj_cholesky import vec_to_tril_matrix, tril_matrix_to_vec
import paddle
import numbers
import numpy as np
paddle.enable_static()

np.random.seed(2024)
paddle.seed(2024)

vocab = {0:'onion', 1:'cvine'}

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'dim', 'concentration', 'sample_method'),
    [
        # ('3-onion-multi-concentration', 
        #  3, 
        #  parameterize.xrand(
        #         (1, 2),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "onion"
        # ),
        # ('3-cvine-multi-concentration', 
        #  3, 
        #  parameterize.xrand(
        #         (1, 2),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "cvine"
        # ),
        # ('2-onion', 
        #  2, 
        #  parameterize.xrand(
        #         (1, ),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "onion"
        # ),
        # ('2-cvine', 
        #  2, 
        #  parameterize.xrand(
        #         (1, ),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "cvine"
        # ),
        ('3-onion', 
         3, 
         parameterize.xrand(
                (1, ),
                dtype='float32',
                max=1.0,
                min=0,
            ), 
         0
        ),
        # ('3-cvine', 
        #  3, 
        #  parameterize.xrand(
        #         (1, ),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "cvine"
        # ),
        # ('10-onion', 
        #  10, 
        #  parameterize.xrand(
        #         (1, ),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "onion"
        # ),
        # ('10-cvine', 
        #  10, 
        #  parameterize.xrand(
        #         (1, ),
        #         dtype='float32',
        #         max=1.0,
        #         min=0,
        #     ), 
        #  "cvine"
        # ),
    ]
)
class TestLKJCholeskyShape(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            dim = paddle.static.data('dim', (), 'int')
            concentration = paddle.static.data('concentration', self.concentration.shape, self.concentration.dtype)[0]
            sample_method = paddle.static.data('sample_method', (), 'int')
            if sample_method == 0:
                sample_method = 'onion'
            elif sample_method == 1:
                sample_method = 'cvine'

            self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(dim, concentration, sample_method)
            self.feeds = {
                'dim' : self.dim,
                'concentration' : self.concentration,
                'sample_method' : self.sample_method
            }
            
    def test_sample_shape(self):
        if isinstance(self.concentration, np.ndarray) and len(self.concentration) > 1:
            extra_shape = (len(self.concentration), self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim)
        else:
            extra_shape = (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim)
        cases = [
            {
                'input': (),
                'expect': ()  + extra_shape,
            },
            # {
            #     'input': (4, 2),
            #     'expect': (4, 2) + extra_shape,
            # },
        ]
        for case in cases:
            with paddle.static.program_guard(self.program):
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.sample(case.get('input')),
                )
            self.assertTrue(tuple(self._paddle_lkj_cholesky.sample(case.get('input')).shape) == case.get('expect'))

# @parameterize.place(config.DEVICES)
# @parameterize.parameterize_cls(
#     (parameterize.TEST_CASE_NAME, 'dim', 'concentration', 'sample_method'),
#     [
#         ('2-onion', 2, 1.0, "onion"),
#         ('2-cvine', 2, 1.0, "cvine"),
#         ('3-onion', 3, 1.0, "onion"),
#         ('3-cvine', 3, 1.0, "cvine"),
#         ('4-onion', 4, 1.0, "onion"),
#         ('4-cvine', 4, 1.0, "cvine"),
#         ('5-onion', 5, 1.0, "onion"),
#         ('5-cvine', 5, 1.0, "cvine"),
#     ]
# )
# class TestLKJCholeskyLogProb(unittest.TestCase):
#     def setUp(self):
#         dim = self.dim
#         concentration = self.concentration
#         sample_method = self.sample_method
#         self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(dim, concentration, sample_method)
                
#     def test_log_prob(self):
#         log_probs = []
#         for i in range(2):
#             sample = self._paddle_lkj_cholesky.sample()
#             log_prob = self._paddle_lkj_cholesky.log_prob(sample)
#             sample_tril = tril_matrix_to_vec(sample, diag = -1)
#             # log_abs_det_jacobian 
#             logabsdet = []
#             logabsdet.append(self.compute_jacobian(sample_tril))
#             logabsdet = paddle.to_tensor(logabsdet)

#             log_probs.append((log_prob - logabsdet).numpy())
#         max_abs_error = np.max(np.abs(log_probs[0] - log_probs[1]))
#         self.assertAlmostEqual(max_abs_error, 0, places=5)
        
#     def _tril_cholesky_to_tril_corr(self, x):
#         x = x.unsqueeze(-1)
#         x = vec_to_tril_matrix(x, -1)
#         diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
#         x = x + diag
#         return tril_matrix_to_vec(x @ x.T, -1)

#     def compute_jacobian(self, x):
#         if x.stop_gradient is not False:
#             x.stop_gradient = False
#         jacobian_matrix = []
#         outputs = self._tril_cholesky_to_tril_corr(x)
#         for i in range(outputs.shape[0]):
#             grad = paddle.grad(outputs=outputs[i], inputs=x, create_graph=False)[0]
#             jacobian_matrix.append(grad)
#         J = paddle.stack(jacobian_matrix, axis=0)
#         _, logabsdet = paddle.linalg.slogdet(J)
#         return logabsdet
    
if __name__ == '__main__':
    unittest.main()
