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

# paddle.enable_static()

np.random.seed(2024)
paddle.seed(2024)



@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'dim', 'concentration', 'sample_method'),
    [
        ('3-onion', 
         3, 
         parameterize.xrand(
                (1, 2),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
            ), 
         "onion"
        ),
        # ('2-cvine', 2, 1.0, "cvine"),
        # ('3-onion', 3, 1.0, "onion"),
        # ('3-cvine', 3, 1.0, "cvine"),
        # ('10-onion', 10, 1.0, "onion"),
        # ('10-cvine', 10, 1.0, "cvine"),
    ]
)
class TestLKJCholesky(unittest.TestCase):
    def setUp(self):
        dim = self.dim
        concentration = self.concentration[0]
        sample_method = self.sample_method
        print(concentration)
        self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(dim, concentration, sample_method)
                
    def test_sample_shape(self):
        if len(self.concentration[0]) > 1:
            
            extra_shape = (len(self.concentration[0]), self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim)
        else:
            extra_shape = (self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim)
        cases = [
            {
                'input': (),
                'expect': ()  + extra_shape,
            },
            {
                'input': (4, 2),
                'expect': (4, 2) + extra_shape,
            },
        ]
        for case in cases:
            self.assertTrue(tuple(self._paddle_lkj_cholesky.sample(case.get('input')).shape) == case.get('expect'))

           
    def test_log_prob(self):
        log_probs = []
        for i in range(2):
            sample = self._paddle_lkj_cholesky.sample()
            log_prob = self._paddle_lkj_cholesky.log_prob(sample)
            sample_tril = tril_matrix_to_vec(sample, diag = -1)
            
            # log_abs_det_jacobian 
            J = self.compute_jacobian(sample_tril)
            _, logabsdet = paddle.linalg.slogdet(J)
            log_probs.append((log_prob - logabsdet).numpy())
        print(sample)
        self.assertAlmostEqual(log_probs[0], log_probs[1], places = 6)
        
    def _tril_cholesky_to_tril_corr(self, x):
        x = x.unsqueeze(-1)
        x = vec_to_tril_matrix(x, -1)
        diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
        x = x + diag
        return tril_matrix_to_vec(x @ x.T, -1)

    def compute_jacobian(self, x):
        if x.stop_gradient is not False:
            x.stop_gradient = False
        jacobian_matrix = []
        for i in range(len(x)):
            # 计算每个输出相对于整个输入向量的梯度
            grad = paddle.grad(outputs=self._tril_cholesky_to_tril_corr(x)[i], inputs=x, create_graph=False)[0]
            jacobian_matrix.append(grad)
        return paddle.stack(jacobian_matrix, axis=0)
    
if __name__ == '__main__':
    unittest.main()
