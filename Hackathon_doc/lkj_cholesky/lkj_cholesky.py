import paddle
from paddle.distribution import distribution
from paddle.distribution import Beta

class LKJCholesky(distribution.Distribution):
    
    def __init__(self, dim, concentration=1.0):
        
        # need dim > 2, TODO add assert 
        self.dim = dim
        
        self.concentration = paddle.to_tensor(concentration)
        batch_shape = self.concentration.shape
        event_shape = paddle.to_tensor((dim, dim))
        
        
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = paddle.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
        )
        offset = paddle.concat([paddle.zeros((1,), dtype=offset.dtype), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        
        super().__init__(batch_shape, event_shape)
        
    def sample(self, sample_shape=paddle.to_tensor([])):
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = paddle.randn(
            self._extended_shape(sample_shape), dtype=y.dtype
        ).tril(-1)
        u_hypersphere = u_normal / u_normal.norm(axis=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = paddle.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        eps = paddle.finfo(w.dtype).tiny
        diag_elems = paddle.clip(1 - paddle.sum(w**2, axis=-1), min=eps).sqrt()
        w += paddle.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        diag_elems = paddle.diagonal(value, offset=0, axis1=-1, axis2=-2)[..., 1:]
        order = paddle.arange(2, self.dim + 1, dtype=self.concentration.dtype)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = paddle.sum(order * paddle.log(diag_elems), axis=-1)
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = paddle.lgamma(alpha) * dm1
        numerator = paddle.mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term

    def _extended_shape(self, sample_shape):
        if not isinstance(sample_shape, paddle.Tensor):
            sample_shape = paddle.to_tensor(sample_shape)
        # Helper function to compute the extended shape for sampling
        print(sample_shape, self.batch_shape, self.event_shape)
        return paddle.to_tensor(sample_shape + self.batch_shape + self.event_shape)
    
    
if __name__ == '__main__':
    
    l = LKJCholesky(3, 0.5)
    l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix