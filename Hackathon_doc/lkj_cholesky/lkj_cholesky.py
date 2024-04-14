import paddle
from paddle.distribution import distribution



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