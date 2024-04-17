# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
from paddle.distribution import distribution
from paddle.distribution.beta import Beta

__all__ = ["LKJCholesky"]



def mvlgamma(a, p):
    """
    :param a: A scalar or tensor of shape (...,)
    :param p: An integer representing the dimension of the multivariate gamma function
    :return: The result of the multivariate gamma function for each element in a
    """
    p_float = float(p)
    order = paddle.arange(0, p_float, dtype=a.dtype)
    return paddle.sum(paddle.lgamma(a.unsqueeze(-1) - 0.5 * order), axis=-1)

def tril_indices(n, k=0):
    # 生成一个n*n的矩阵
    full_matrix = paddle.ones((n, n), dtype='int32')
    # 生成一个下三角矩阵
    tril_matrix = paddle.tril(full_matrix, diagonal=k)
    # 获取下三角矩阵非零元素的索引
    rows, cols = paddle.nonzero(tril_matrix, as_tuple=True)
    return rows.flatten(), cols.flatten()

def matrix_to_tril(x, diagonal=0):
    matrix_dim = x.shape[-1]
    rows, cols = tril_indices(matrix_dim, diagonal)
    # print(x[..., rows, cols])
    return x[..., rows, cols]

def construct_matrix_lower(p):
    
    # cal dim
    dim = int((math.sqrt(paddle.to_tensor(1 + 8*p.shape[0])) + 1) / 2)
    
    # 创建一个dim x dim的零矩阵
    matrix = paddle.zeros(shape=[dim, dim], dtype='float32')
    # import pdb; pdb.set_trace()
    # 计算下三角矩阵的索引
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    
    # print(lower_indices)
    # 将p中的元素填充到matrix的相应位置
    matrix = paddle.scatter_nd_add(matrix, lower_indices, paddle.flatten(p))
    
    return matrix


class LKJCholesky(distribution.Distribution):
    def __init__(self, dim, concentration=1.0, sample_method="onion"):
        # need dim > 2, TODO add assert
        self.dim = dim
        self.concentration = paddle.to_tensor(concentration)
        self.sample_method = sample_method
        
        batch_shape = self.concentration.shape
        event_shape = (dim, dim)

        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = paddle.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
        )
            
        if sample_method == "onion":
            offset = paddle.concat([paddle.zeros((1,), dtype=offset.dtype), offset])
            beta_conc1 = offset + 0.5
            beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
            self._beta = Beta(beta_conc1, beta_conc0)
        elif sample_method == "cvine":
            offset_tril = matrix_to_tril(paddle.broadcast_to(0.5 * offset, [self.dim - 1,self.dim - 1]))
            beta_conc = marginal_conc.unsqueeze(-1) - offset_tril
            self._beta = Beta(beta_conc, beta_conc)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        # print("check shape", batch_shape, event_shape)
        # import pdb; pdb.set_trace()
        super().__init__(batch_shape, event_shape)

    def _onion(self, sample_shape):
        
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        
        # u ~ N(0, 1)
        u_normal = paddle.randn(
            self._extend_shape(sample_shape), dtype=y.dtype
        ).tril(-1)
        
        u_hypersphere = u_normal / u_normal.norm(axis=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = paddle.sqrt(y) * u_hypersphere
        
        # Fill diagonal elements; clamp for numerical stability
        eps = paddle.finfo(w.dtype).tiny
        diag_elems = paddle.clip(
            1 - paddle.sum(w**2, axis=-1), min=eps
        ).sqrt()
        
        w += paddle.diag_embed(diag_elems)
        return w

    def _cvine(self, sample_shape):
        # print(sample_shape)
        beta_sample = self._beta.sample(sample_shape).unsqueeze(-1)
        partial_correlation = 2 * beta_sample - 1
        # print("cvine \n", partial_correlation)
        
        partial_correlation = construct_matrix_lower(partial_correlation)
        # print("beta_sample", beta_sample)
        # print("partial_correlation", partial_correlation)
        
        eps = paddle.finfo(beta_sample.dtype).tiny
        r = paddle.clip(partial_correlation, min=(-1+eps), max=(1-eps))
        z = r**2
        # print("cvine \n", partial_correlation)
        # print("cvine \n", r)
        # print("cvine \n", z)
        z1m_cumprod_sqrt = paddle.cumprod(paddle.sqrt(1 - z), dim=-1)
        pad_width = [0, 0] * (z1m_cumprod_sqrt.ndim - 1) + [1, 0]
        z1m_cumprod_sqrt_shifted = paddle.nn.functional.pad(
            z1m_cumprod_sqrt[..., :-1],  # 选择除了最后一个元素之外的所有元素
            pad=pad_width,
            mode="constant",
            value=1.0  # 指定填充的常数值为1.0
        )
        # print("cvine \n", paddle.sqrt(1 - z))
        # print("cvine \n", z1m_cumprod_sqrt)
        # print("cvine \n", z1m_cumprod_sqrt_shifted)
        # print("cvine \n", r)
        r += paddle.eye(partial_correlation.shape[-2], partial_correlation.shape[-1])
        return r * z1m_cumprod_sqrt_shifted
        # raise ValueError("`cvine` not impl yet.")

    
    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = paddle.to_tensor([])
        if self.sample_method == "onion":
            return self._onion(sample_shape)
        else:
            return self._cvine(sample_shape)
        

    def log_prob(self, value):
        diag_elems = paddle.diagonal(value, offset=0, axis1=-1, axis2=-2)[
            ..., 1:
        ]
        order = paddle.arange(2, self.dim + 1, dtype=self.concentration.dtype)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order

        unnormalized_log_pdf = paddle.sum(
            order * paddle.log(diag_elems), axis=-1
        )
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1

        alpha = self.concentration + 0.5 * dm1
        denominator = paddle.lgamma(alpha) * dm1
        numerator = mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)

        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term

