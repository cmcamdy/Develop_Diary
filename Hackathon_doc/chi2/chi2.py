import paddle
from paddle.distribution import Gamma


class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:df.
    This is exactly equivalent to Gamma(concentration=0.5*df, rate=0.5)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(paddle.to_tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """
    def __init__(self, df):
        if not isinstance(df, paddle.Tensor):
            df = paddle.to_tensor(df)
        super().__init__(0.5 * df, paddle.to_tensor(0.5))
    
    @property
    def df(self):
        return self.concentration * 2
    